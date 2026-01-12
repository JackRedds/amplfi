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

run_type = "multi_sg"
config_path = f"/projects/bcse/jredepenning/amplfi/runs/{run_type}/sg.yaml"
ckpt_path = f"/projects/bcse/jredepenning/amplfi-outdir/sg_runs/{run_type}/train_logs/best.ckpt"



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

    learning_rate = [1e-4, 3e-4, 1e-3]
    transforms = [12, 16, 20]
    hidden_features = [[256,256,256], [512,512,512]]
    weight_decay = [0.0, 4e-4]

    # Refinement
    # pct_start = [0.1, 0.2, 0.3]
    # freq_context_dim = [128, 256]
    # time_context_dim = [8, 16]
 
    learning_rate = trial.suggest_categorical("learning_rate", learning_rate)
    transforms = trial.suggest_categorical("transforms", transforms)
    hidden_features = trial.suggest_categorical("hidden_features", hidden_features)
    weight_decay = trial.suggest_categorical("weight_decay", weight_decay)

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
        patience=30,
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
    model.hparams["hidden_features"] = hidden_features
    model.hparams["transforms"] = transforms

    train_dataloader, val_dataloader = prepare_data(datamodule)
    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics['valid_loss'].item()

if __name__ == '__main__':
    n_trials = 20
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    df.to_json('hyperparameters.json')