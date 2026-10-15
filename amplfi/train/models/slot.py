from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from ml4gw.transforms import ChannelWiseScaler

if TYPE_CHECKING:
    from ...utils.result import AmplfiResult
    from ..architectures.slots import SlotHead

Tensor = torch.Tensor


class SlotModelMixin:
    """
    Mixin for `FlowModel` and `PosteriorSamplerModel` adding support
    for architectures with a `SlotHead`, which also infer the number
    of components of a signal and each component's parameters.
    Requires `amplfi.train.data.datasets.SlotFlowDataset`.

    When the architecture has no slot head, models behave
    exactly as without this mixin.

    When testing, the global parameters are analyzed as usual,
    and the slot samples are added to each result's posterior as
    `n_components` and `{param}_{i}` for the `i`th slot (NaN when the
    sample has fewer than `i + 1` components). Slot indices are
    arbitrary, so they don't correspond to the injection's component
    order, which is stored in the injection parameters the same way.
    The predicted probabilities of each number of components are
    stored in the result's `meta_data["count_probs"]`. If
    `cos_phase` and `sin_phase` are slot parameters, the
    recovered `phase_{i}` in `[0, 2 pi)` is added as well
    """

    @property
    def slot_head(self) -> Optional["SlotHead"]:
        return getattr(self.model, "slot_head", None)

    def init_slots(self):
        if self.slot_head is not None:
            # unfit scaler so that it is saved to
            # and loaded from checkpoints
            self.slot_scaler = ChannelWiseScaler(
                self.slot_head.num_slot_params
            )

    def setup(self, stage):
        super().setup(stage)
        datamodule = self.trainer.datamodule
        slot_params = getattr(datamodule, "slot_params", None)
        if self.slot_head is None:
            if slot_params is not None:
                raise ValueError(
                    "The dataset returns per-component parameters, "
                    "but the architecture has no `slot_head`"
                )
            return

        if slot_params is None:
            raise ValueError(
                "The architecture has a `slot_head`, which requires "
                "a `SlotFlowDataset`"
            )
        if list(slot_params) != list(self.slot_head.slot_params):
            raise ValueError(
                f"Dataset slot_params {slot_params} don't match the "
                f"slot head's {self.slot_head.slot_params}"
            )
        if datamodule.max_components != self.slot_head.max_components:
            raise ValueError(
                f"Dataset max_components {datamodule.max_components} "
                f"doesn't match the slot head's "
                f"{self.slot_head.max_components}"
            )
        self.slot_scaler = datamodule.slot_scaler

    # ================================================ #
    # Training
    # ================================================ #

    def slot_step(self, batch, stage: str) -> Tensor:
        strain, asds, targets, _ = batch
        losses = self.model.slot_losses((strain, asds), targets)
        loss = losses.pop("loss").mean()
        losses["count_accuracy"] = losses.pop("count_correct")

        on_step = stage == "train"
        for k, v in losses.items():
            self.log(
                f"{stage}_{k}",
                v.mean(),
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
                logger=True,
            )
        # `valid_loss` is monitored by checkpointing and the scheduler
        self.log(
            f"{stage}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=stage == "valid",
            sync_dist=True,
            logger=True,
        )
        return loss

    # ================================================ #
    # Inference
    # ================================================ #

    def sample_posterior(
        self, n: int, context
    ) -> tuple[Tensor, Optional[dict[str, Tensor]]]:
        """
        Draw `n` (scaled) samples of the global parameters and,
        if there is a slot head, of the slots
        """
        if self.slot_head is None:
            return self.model.sample(n, context=context), None
        return self.model.sample_with_slots(n, context)

    def descale_slots(self, slots: Tensor, mask: Tensor) -> Tensor:
        """De-scale slots `(..., K, S)`, setting padded slots to NaN"""
        shape = slots.shape
        slots = self.slot_scaler(slots.reshape(-1, shape[-1]).T, reverse=True)
        slots = slots.T.reshape(shape)
        nan = torch.full_like(slots, float("nan"))
        return torch.where(mask[..., None], slots, nan)

    def add_slot_results(
        self,
        result: "AmplfiResult",
        slot_samples: dict[str, Tensor],
        keep: Tensor,
        targets: Optional[dict[str, Tensor]] = None,
    ):
        """
        Add the slot samples of a single event to `result`.
        `keep` masks the samples kept after filtering the global
        parameters, and `targets` holds the injected slots
        """
        keep = keep.to(slot_samples["slots"].device)
        mask = slot_samples["mask"][:, 0][keep]
        slots = self.descale_slots(slot_samples["slots"][:, 0][keep], mask)
        slots = slots.cpu().numpy()

        posterior = {
            "n_components": slot_samples["n_components"][:, 0][keep]
        }
        posterior["n_components"] = posterior["n_components"].cpu().numpy()
        for i in range(self.slot_head.max_components):
            for j, param in enumerate(self.slot_head.slot_params):
                posterior[f"{param}_{i}"] = slots[:, i, j]
        self.add_phase(posterior)
        for k, v in posterior.items():
            result.posterior[k] = v

        if result.meta_data is None:
            result.meta_data = {}
        count_probs = slot_samples["count_probs"][0].cpu().numpy()
        result.meta_data["count_probs"] = count_probs.tolist()

        if targets is not None and result.injection_parameters is not None:
            true_mask = targets["mask"][0]
            true_slots = self.descale_slots(targets["slots"][0], true_mask)
            true_slots = true_slots.cpu().numpy()
            injection = result.injection_parameters
            injection["n_components"] = int(true_mask.sum())
            for i in range(self.slot_head.max_components):
                for j, param in enumerate(self.slot_head.slot_params):
                    injection[f"{param}_{i}"] = float(true_slots[i, j])
            self.add_phase(injection)

    def add_phase(self, columns: dict):
        """
        Add `phase_{i}` to per-slot `columns` when the
        phase is inferred as `cos_phase` and `sin_phase`
        """
        slot_params = self.slot_head.slot_params
        if "cos_phase" not in slot_params or "sin_phase" not in slot_params:
            return
        for i in range(self.slot_head.max_components):
            phase = np.arctan2(
                columns[f"sin_phase_{i}"], columns[f"cos_phase_{i}"]
            )
            columns[f"phase_{i}"] = np.mod(phase, 2 * np.pi)

    @staticmethod
    def split_targets(
        parameters,
    ) -> tuple[Optional[Tensor], Optional[dict[str, Tensor]]]:
        """
        Split test parameters into the global parameters
        and, from a `SlotFlowDataset`, the slot targets
        """
        if isinstance(parameters, dict):
            return parameters["global"], parameters
        return parameters, None
