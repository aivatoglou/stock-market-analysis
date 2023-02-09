import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import wandb

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)

from src.models.net import make_network
from src.models.train import train_net
from src.utils.early_stopping import EarlyStopper

with open("src/config.json") as file:
    init_config = json.load(file)

WINDOW = init_config["WINDOW"]
SEED = init_config["SEED"]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

hyperparameter_defaults = dict(
    learning_rate=init_config["DEFAULT_LEARNING_RATE"],
    batch_size=init_config["DEFAULT_BATCH_SIZE"],
    width=init_config["DEFAULT_WIDTH"],
    layers=init_config["DEFAULT_LAYERS"],
)
wandb.init(project="stock-predictor", entity="johndooe", config=hyperparameter_defaults)
wandb_config = wandb.config

#!---------------------------- READ DATA -----------------------------------!#
df = pd.read_csv(f"{init_config['DATA_DIR']}")
df = df[["Date", "Close"]]
print(f"Original shape {df.shape}")

if init_config["INTERPOLATION"]:
    #!------------ INTERPOLATE WEEKENDS & Public Holidays -----------------------!#
    df.index = pd.to_datetime(df["Date"])
    dates = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(dates, fill_value=0)
    df = df.drop("Date", axis=1)
    df.replace(0, np.NaN, inplace=True)
    df = df.interpolate()
    print(f"Interpolated shape {df.shape}")

#!---------------------------- CREATE WINDOWS -----------------------------------!#
X = []
y = []

for i in range(WINDOW, len(df)):

    if WINDOW + i < len(df):
        X.append(df["Close"].iloc[i - WINDOW : i].values.tolist())
        y.append(df["Close"].iloc[i : WINDOW + i].values.tolist())
    else:
        break


_X = np.array(X)
_y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=init_config["TEST_SET_SLICE"], random_state=SEED)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_test, y_test, test_size=init_config["VALIDATION_SET_SLICE"], random_state=SEED
)

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)

X_valid = torch.Tensor(X_valid)
y_valid = torch.Tensor(y_valid)

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

#!-----------------------------------------------------------------------------------------------!#
#!------------------------------------------TUNING-----------------------------------------------!#


def main():

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=wandb_config.batch_size, drop_last=True, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=wandb_config.batch_size, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb_config.batch_size, drop_last=True)

    net = make_network(input_dim=WINDOW, width=wandb_config.width, L=wandb_config.layers, out_features=WINDOW)
    if torch.cuda.is_available():
        net = net.cuda()

    # Compile net
    # opt_net = torch.compile(net)

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=wandb_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
    early_stopper = EarlyStopper(patience=5, min_delta=0)

    print(
        f"batch size: {wandb_config.batch_size} lr: {wandb_config.learning_rate} width: {wandb_config.width} n_layers: {wandb_config.layers} \n"
    )
    #!--------------- Training process ---------------!#
    for epoch in range(init_config["EPOCHS"] + 1):

        loss, vloss, y_pred, y_true = train_net(
            net, optimizer, loss_function, scheduler, train_dataloader, valid_dataloader, test_dataloader
        )

        if epoch % 10 == 0:

            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            score = r2_score(y_true, y_pred)
            wandb.log({"r2_score": score})

            print(
                f"Epoch {epoch} \t Training Loss: {loss:.5f} \t Validation Loss: {vloss:.5f} \t Test-set R2 score {score}"
            )

            if early_stopper.early_stop(vloss):
                print("-" * 70)
                print(
                    f"Epoch {epoch} \t Training Loss: {loss:.5f} \t Validation Loss: {vloss:.5f} \t Test-set R2 score {score} \n"
                )
                break

        # Wandb logs
        wandb.log({"epoch_training_loss": loss})
        wandb.log({"epoch_validation_loss": vloss})

    #!--------------- EXPORT ONNX ---------------!#
    net.train(False)
    x = torch.randn(wandb_config.batch_size, WINDOW, requires_grad=True).cuda()
    torch_out = net(x)
    torch.onnx.export(
        net,
        x,
        f"models/net_{wandb_config.batch_size}_{wandb_config.learning_rate}_{wandb_config.width}_{wandb_config.layers}.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "features"}, "output": {0: "batch_size", 1: "features"}},
        verbose=False,
    )


if __name__ == "__main__":
    main()
