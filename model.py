
import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.go = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.go(x)
        return logits


def train(model, epochs, dataloader, loss_fn, optimizer):
    train_loss = []
    for t in range(epochs):
        model.train()
        epoch_loss = []
        for b, data in enumerate(dataloader):
            # xi, xo = xi.to(torch.device("cpu")), xo.to(torch.device("cpu"))
            xi, xo = data[0], data[1]
            # print(xi.shape)

            x_pred = model(xi)
            loss = loss_fn(x_pred, xo)
            epoch_loss.append(loss.item())

            print(f"Epoch {t} | Batch {b} | Loss {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss.append(np.array(epoch_loss).mean())

def evaluate(model, dataloader):

    preds = []
    for b, data in enumerate(dataloader):
        x_in = data[0]
        model.eval()
        with torch.no_grad():
            pred = model(x_in)

        preds.extend(np.array(pred[:,0]))

    return np.array(preds)