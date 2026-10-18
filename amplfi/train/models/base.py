import logging
import math
import sys
from pathlib import Path
from typing import Optional
import lightning.pytorch as pl
import numpy as np
import torch
from ml4gw.transforms import ChannelWiseScaler
from amplfi.train.callbacks import SaveWandbUrl

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution


class AmplfiModel(pl.LightningModule):
    """
    Amplfi model base class

    Encodes common functionality for all models,
    such as on-device augmentation and preprocessing,

    Args:
        checkpoint:
            Path to a model checkpoint to load. This will load in weights
            for both flow and embedding. Should only be specified when
            running `trainer.test`. For resuming a `trainer.fit` run from
            a checkpoint, use the --ckpt_path `Trainer` argument.
    """

    def __init__(
        self,
        inference_params: list[str],
        learning_rate: float,
        pct_start: float,
        train_outdir: Path,
        pretrained_path: str = None,
        strict_loading: bool = False,
        test_outdir: Optional[Path] = None,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self._logger = self.init_logging(verbose)

        if test_outdir is None:
            test_outdir = train_outdir / "test_results"

        self.test_outdir = test_outdir
        self.train_outdir = train_outdir

        self.inference_params = inference_params

        self.save_hyperparameters()

        # initialize an unfit scaler here so that it is available
        # for the LightningModule to save and load from checkpoints
        self.scaler = ChannelWiseScaler(len(inference_params))

    def on_train_start(self):
        self.train_outdir.mkdir(exist_ok=True, parents=True)

    def init_logging(self, verbose):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )

        world_size, rank = self.get_world_size_and_rank()
        logger_name = self.__class__.__name__
        if world_size > 1:
            logger_name += f":{rank}"
        return logging.getLogger(logger_name)

    def get_world_size_and_rank(self) -> tuple[int, int]:
        """
        Name says it all, but generalizes to the case
        where we aren't running distributed training.
        """
        if not torch.distributed.is_initialized():
            return 1, 0
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return world_size, rank

    def setup(self, stage):
        if self.hparams.pretrained_path and stage == "fit":
            self._logger.info(
                "Pretrained path provided as {}".format(
                    self.hparams.pretrained_path
                )
            )
            self._logger.info(
                "Attempting to restore model weights. "
                "Note that this overrides checkpoint provided "
                "using --ckpt_path to the Lightning CLI."
            )
            checkpoint = torch.load(
                self.hparams.pretrained_path, map_location=self.device
            )
            self.load_state_dict(
                checkpoint["state_dict"], strict=self.hparams.strict_loading
            )
            self._logger.info("Model weights restored from pretrained path.")
        self.scaler = self.trainer.datamodule.scaler

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * math.sqrt(world_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            pct_start=self.hparams.pct_start,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "step",
            },
        }

    def scale(self, parameters, reverse: bool = False) -> Tensor:
        """
        Apply standard scaling to transformed parameters
        """
        parameters = parameters.transpose(1, 0)
        scaled = self.scaler(parameters, reverse=reverse)
        scaled = scaled.transpose(1, 0)
        return scaled

    # integer-valued parameters that are dequantized
    # with `amplfi.train.prior.dequantize` during training
    discrete_params = ["n_components"]

    def requantize(self, parameters: Tensor) -> Tensor:
        """
        Floor descaled samples of discrete inference parameters
        to recover integer values. Lower values are clamped to 1,
        since dequantization only ever adds noise in [0, 1)
        """
        parameters = parameters.clone()
        for i, param in enumerate(self.inference_params):
            if param in self.discrete_params:
                parameters[:, i] = parameters[:, i].floor().clamp_min(1)
        return parameters

    def log_prior(self, samples: dict[str, Tensor]) -> Tensor:
        """
        Log probability of posterior samples under the training prior.
        This is NaN when the waveform sampler's prior doesn't define
        every inference parameter, e.g. when testing on waveforms from
        a different family than the model was trained on
        """
        try:
            return self.training_prior.log_prob(samples)
        except KeyError as e:
            if not getattr(self, "_warned_log_prior", False):
                self._logger.warning(
                    f"Training prior doesn't define parameter {e}; "
                    "setting log prior of posterior samples to NaN"
                )
                self._warned_log_prior = True
            first = next(iter(samples.values()))
            return torch.full((len(first),), float("nan"))

    @staticmethod
    def drop_missing(injection_parameters: dict[str, float]):
        """
        Remove parameters without a true value (NaN), which the
        test waveforms didn't define
        """
        return {
            k: v for k, v in injection_parameters.items() if not np.isnan(v)
        }

    def configure_callbacks(self):
        callbacks = [SaveWandbUrl()]
        return callbacks
