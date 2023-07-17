import torch


class EarlyStopping:
    def __init__(self, name, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.attention_loss_min = float("inf")
        self.name = name

    def __call__(self, attention_loss, model):
        score = -attention_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(attention_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(attention_loss, model)
            self.counter = 0

    def save_checkpoint(self, attention_loss, model):
        """Saves model when attention loss decreases."""
        if attention_loss <= self.attention_loss_min:
            print(
                f"Test loss decreased ({self.attention_loss_min:.6f} --> {attention_loss:.6f}).  Saving model ..."
            )
            torch.save(
                model.state_dict(),
                self.name.split(".pt")[0] + f" best.pt",
            )
            self.attention_loss_min = attention_loss
