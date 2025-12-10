import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
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

    # Architecture:
    #   num_layers: 1 -> 5
    #   hidden_size: 32 -> 1024
    # cli.model.hparams['arch']['init_args']['hidden_features']

    
    batch_size_choices = [128, 256, 512]
    batches_per_epoch_choices = [100, 150, 200]
    batch_size = trial.suggest_categorical("batch_size", batch_size_choices)
    batches_per_epoch = trial.suggest_categorical("batches_per_epoch", batches_per_epoch_choices)
    learning_rate = trial.suggest_float("learning_rate", 7.14e-4, 7.14e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 4.2e-4, 4.2e-2, log=True)

    # batch_size = 256
    # learning_rate = 7.14e-3
    weight_decay = 4.2e-3
    # batches_per_epoch = 120

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

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        verbose=False,
        mode='min'
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='valid_loss')

    model = cli.model
    datamodule = cli.datamodule
    trainer = cli.trainer
    trainer.callbacks.append(early_stop_callback)
    trainer.callbacks.append(pruning_callback)
    # model.eval();
    datamodule.transforms_to_device()
    datamodule.setup("fit")
    # trainer.max_epochs = max_epochs
    model.hparams["learning_rate"] = learning_rate
    model.hparams["weight_decay"] = weight_decay
    datamodule.hparams["batch_size"] = batch_size
    datamodule.hparams["batches_per_epoch"] = batches_per_epoch

    train_dataloader, val_dataloader = prepare_data(datamodule)
    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics['valid_loss'].item()

if __name__ == '__main__':
    n_trials = 10
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    df.to_json('hyperparameters.json')