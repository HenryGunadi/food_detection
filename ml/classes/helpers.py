from pathlib import Path
import copy
import torch.nn as nn
import torch

class EarlyStopping():
    def __init__(self, model: nn.Module, save_path: Path, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta # minimum change in the monitored metric
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.model = model
        self.save_path = save_path / "best_model.pth"

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0

        else:
            self.no_improvement_count += 1
            if self.no_improvement_count == self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed")
                    print("Best Loss", self.best_loss)

                best_state_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_state_model, self.save_path)
                print("The best model is saved.")