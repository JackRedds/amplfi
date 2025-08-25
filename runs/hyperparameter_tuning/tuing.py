import torch
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
import optuna
from amplfi.train.data.datasets.base import AmplfiDataset
from amplfi.train.models.base import AmplfiModel
from amplfi.train.cli.flow import AmplfiFlowCLI
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

config_path = "/projects/bcse/jredepenning/amplfi/runs/sg_sky_loc/sg.yaml"
ckpt_path = "/projects/bcse/jredepenning/amplfi-outdir/run5/sg_sky_loc/train_logs/best.ckpt"

cli = AmplfiFlowCLI(
        AmplfiModel,
        AmplfiDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=[
        "test",             # Subcommand
        "--config", config_path,
        "--ckpt_path", ckpt_path,
        ]
)

def prepare_data(datamodule):
    datamodule.transforms_to_device()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    train_batch = []
    val_batch = []
    datamodule.trainer.training = True
    for batch in tqdm(train_dataloader):
        t_batch = datamodule.on_after_batch_transfer(batch, device)
        train_batch.append(t_batch)
    
    datamodule.trainer.training = False
    datamodule.trainer.validating = True
    for batch in tqdm(val_dataloader):
        v_batch = datamodule.on_after_batch_transfer(batch, device)
        val_batch.append(v_batch)

    train_dataloader = DataLoader(train_batch, batch_size=None)
    val_dataloader = DataLoader(val_batch, batch_size=None)

    return train_dataloader, val_dataloader


def objective(trial):
    batch_size = trial.suggest_int("batch_size", 128, 1024, log=True)
    learning_rate = trial.suggest_float("learning_rate", 7.14e-5, 7.14e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 4.2e3, 4.2e5, log=True)
    batches_per_epoch = trial.suggest_int("batches_per_epoch", 2, 5)
    model = cli.model
    model.eval();
    datamodule = cli.datamodule
    datamodule.transforms_to_device()
    datamodule.setup("fit")
    model.hparams["learning_rate"] = learning_rate
    model.hparams["weight_decay"] = weight_decay
    datamodule.hparams["batch_size"] = batch_size
    datamodule.hparams["batches_per_epoch"] = batches_per_epoch

    train_dataloader, val_dataloader = prepare_data(datamodule)

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    trainer = cli.trainer
    trainer.callbacks = [pruning_callback]
    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics['val_loss'].item()


n_trials = 10
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)