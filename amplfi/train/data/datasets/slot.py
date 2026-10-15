import torch
from ml4gw.transforms import ChannelWiseScaler

from .flow import FlowDataset

Tensor = torch.Tensor


class SlotFlowDataset(FlowDataset):
    """
    Lightning DataModule for training and testing models whose
    architecture has a `SlotHead`, on waveforms with a variable
    number of components (see `amplfi.train.architectures.slots`).
    Works with `FlowModel`, `DiffusionModel` and `FlowMatchingModel`.

    `inference_params` are the event-level (global) parameters,
    exactly as for `FlowDataset`. In addition, the per-component
    parameters listed in `slot_params` are returned for every
    component, along with a mask of which components are real.

    The waveform sampler's prior must return per-component
    parameters of shape `(N, max_components)` and a boolean
    `component_mask`, e.g. `amplfi.train.prior.AmplfiMultiWFPrior`
    with `max_components` set.

    Batches passed to the model have parameters of the form
    `{"global": (B, P), "slots": (B, K, S), "mask": (B, K)}`,
    where `K = max_components` and `S = len(slot_params)`.
    Global and slot parameters are transformed and standard scaled,
    and padded slots are set to zero.

    Args:
        *args:
            See arguments in `amplfi.train.data.datasets.FlowDataset`
        slot_params:
            Per-component parameters to perform inference on
        max_components:
            Maximum number of components per event
    """

    def __init__(
        self,
        *args,
        slot_params: list[str],
        max_components: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.slot_params = slot_params
        self.max_components = max_components

    @property
    def num_slot_params(self):
        return len(self.slot_params)

    # ================================================ #
    # Slot parameter helpers
    # ================================================ #

    def stack_slots(self, parameters: dict[str, Tensor]) -> Tensor:
        """Stack per-component parameters into a `(N, K, S)` tensor"""
        slots = torch.stack(
            [torch.as_tensor(parameters[k]) for k in self.slot_params], dim=-1
        )
        if slots.shape[1] != self.max_components:
            raise ValueError(
                f"Expected per-component parameters padded to "
                f"{self.max_components} components, got {slots.shape[1]}. "
                "Set `max_components` on the prior to match the dataset"
            )
        return slots

    def transform_slots(self, slots: Tensor, mask: Tensor) -> Tensor:
        """Transform and scale `(N, K, S)` slots, zeroing padded slots"""
        transformed = self.transform(
            {k: slots[..., i] for i, k in enumerate(self.slot_params)}
        )
        slots = torch.stack(
            [transformed[k] for k in self.slot_params], dim=-1
        )
        # padded slots can be non-finite after
        # transforms (e.g. the log of zero hrss)
        slots = torch.where(mask[..., None], slots, torch.zeros_like(slots))
        shape = slots.shape
        slots = self.slot_scaler(slots.reshape(-1, shape[-1]).T).T
        slots = slots.reshape(shape)
        return torch.where(mask[..., None], slots, torch.zeros_like(slots))

    def fit_slot_scaler(self) -> ChannelWiseScaler:
        scaler = ChannelWiseScaler(self.num_slot_params)
        parameters = self.waveform_sampler.get_fit_parameters()
        mask = parameters["component_mask"]
        slots = self.stack_slots(parameters)
        transformed = self.transform(
            {k: slots[..., i] for i, k in enumerate(self.slot_params)}
        )
        # only fit to real components
        fit = torch.stack([transformed[k][mask] for k in self.slot_params])
        scaler.fit(fit)
        return scaler

    # ================================================ #
    # Overrides of AmplfiDataset / FlowDataset
    # ================================================ #

    def build_transforms(self, stage):
        super().build_transforms(stage)
        model = self.trainer.model
        if stage in ["predict", "test"] and model.slot_scaler.built:
            self._logger.info("Using pre-fit slot scaler")
            self.slot_scaler = model.slot_scaler
        else:
            self._logger.info("Fitting slot scaler to slot parameters")
            self.slot_scaler = self.fit_slot_scaler()

    def setup(self, stage: str) -> None:
        if stage not in ["fit", "validate", "test"]:
            raise NotImplementedError(
                f"SlotFlowDataset does not support stage '{stage}'"
            )
        super().setup(stage)

        # append flattened slot parameters and the component
        # mask to the stacked validation/testing parameters,
        # so that they're batched alongside the waveforms
        if stage == "test":
            parameters = self.test_waveform_parameters
            stacked = self.test_inference_params
        else:
            parameters = self.val_parameters
            stacked = self.val_inference_params

        mask = parameters["component_mask"]
        slots = self.stack_slots(parameters).float()
        self.num_global_params = stacked.shape[-1]
        stacked = torch.column_stack(
            [stacked, slots.reshape(len(slots), -1), mask.float()]
        )

        if stage == "test":
            self.test_inference_params = stacked
            # save every per-component parameter of the injections
            # (not only slot_params, e.g. also the raw phase) alongside
            # the other injection parameters, NaN for padded components
            for k, v in parameters.items():
                if k == "component_mask" or v.ndim != 2:
                    continue
                v = torch.where(mask, v.float(), float("nan"))
                for i in range(self.max_components):
                    self.test_parameters[f"{k}_{i}"] = v[:, i].numpy()
        else:
            self.val_inference_params = stacked

    def unpack_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        num_global = self.num_global_params
        K, S = self.max_components, self.num_slot_params
        unpacked = super().unpack_parameters(parameters[:, :num_global])

        slots = parameters[:, num_global : num_global + K * S]
        slots = slots.reshape(-1, K, S)
        unpacked.update(
            {k: slots[..., i] for i, k in enumerate(self.slot_params)}
        )
        unpacked["component_mask"] = parameters[:, num_global + K * S :] > 0.5
        return unpacked

    def inject(
        self,
        X: Tensor,
        cross: Tensor,
        plus: Tensor,
        parameters: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
        mask = parameters["component_mask"].bool()
        slots = self.stack_slots(parameters)

        X, asds, global_params, snrs = super().inject(
            X, cross, plus, parameters
        )
        slots = self.transform_slots(slots.to(X.device), mask.to(X.device))

        targets = {"global": global_params, "slots": slots, "mask": mask}
        return X, asds, targets, snrs
