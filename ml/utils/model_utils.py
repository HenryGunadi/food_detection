from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from classes.helpers import EarlyStopping
from pathlib import Path
import os

def train_step(dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, model: nn.Module, device: str = "cuda"):
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred_logits = model(X)

        # loss
        loss = loss_fn(y_pred_logits, y) # (pred, true)
        train_loss += loss.item() # converts Pytorch tensor into plain python number type

        # optimizer zero grad
        optimizer.zero_grad()

        # backprop
        loss.backward()

        # gradient descent
        optimizer.step()

        # we dont use softmax because we're using crossentropyloss which already applies softmax internally
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: nn.Module, loss_fn: torch.nn.Module, dataloader: DataLoader, device: str = "cuda"):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # forward pass 
            test_pred_logits = model(X)

            # loss
            loss = loss_fn(test_pred_logits, y) # (pred, true)
            test_loss += loss.item()

            test_pred_label = torch.argmax(test_pred_logits, dim=1)
            test_acc += ((test_pred_label == y).sum().item() / len(test_pred_label))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def valid_step(model: nn.Module, loss_fn: torch.nn.Module, dataloader: DataLoader, device: str = "cuda"):
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # converts to the correct device
            X, y = X.to(device), y.to(device)

            # forward
            val_pred_logits = model(X)

            # calculate loss and acc
            loss = loss_fn(val_pred_logits, y) # (pred, true)
            val_loss += loss.item()
            
            val_pred = torch.argmax(val_pred_logits, dim=1)
            val_acc += ((val_pred == y).sum().item() / len(val_pred))

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    return val_loss, val_acc
            
def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          train_dataloader: DataLoader,
          validation_dataloader: DataLoader,
          n_epochs: int,
          scheduler: torch.optim = None,
          early_stopping: EarlyStopping = None,
          device: str = "cuda"):
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc = train_step(dataloader=train_dataloader,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        model=model,
                                        device=device)
        val_loss, val_acc = valid_step(dataloader=validation_dataloader,
                                        loss_fn=loss_fn,
                                        model=model,
                                        device=device)
        
        print(f"EPOCH : {epoch + 1}\nTrain Loss : {train_loss} | Train Acc : {train_acc}\nValidation Loss : {val_loss} | Validation Acc : {val_acc}")
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        if early_stopping is not None:
            # Check early stopping condition
            early_stopping.check_early_stop(val_loss)
            
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if scheduler is not None:
            scheduler.step(val_loss)

    return results