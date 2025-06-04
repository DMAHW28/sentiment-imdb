from abc import ABC, abstractmethod
import torch

class TrainerType(ABC):
    def __init__(self, model, criterion, optimizer, device='mps' if torch.backends.mps.is_available() else 'cpu', without_mask = False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.without_mask = without_mask
        self.device = torch.device(device)
        self.eval_device = torch.device('cpu') if not without_mask else self.device

        # Training statistics
        self.train_loss = 0
        self.val_loss = 0
        self.train_acc = 0
        self.val_acc = 0

    def init_stats_params(self):
        "Reset training statistics"
        self.train_loss = 0
        self.val_loss = 0
        self.train_acc = 0
        self.val_acc = 0

    @abstractmethod
    def train_step(self, x, y, mask=None):
        "Training step"
        pass

    @abstractmethod
    def val_step(self, x, y, mask=None):
        "Validation step"
        pass

class Trainer(TrainerType):
    def __init__(self, model, criterion, optimizer, device='mps' if torch.backends.mps.is_available() else 'cpu', without_mask = False):
        super(Trainer, self).__init__(model, criterion, optimizer, device, without_mask)

    def train_step(self, x, y, mask=None):
        "Training step"
        self.model.train()
        self.model.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        # compute loss
        self.train_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.train_acc += correct

    def val_step(self, x, y, mask=None):
        "Validation step"
        self.model.eval()
        self.model.to(self.eval_device)
        x, y = x.to(self.eval_device), y.to(self.eval_device)
        if mask is not None:
            mask = mask.to(self.eval_device)

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred, y)
        # compute loss
        self.val_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.val_acc += correct


class TrainerBERT(TrainerType):
    def __init__(self, model, criterion, optimizer, device='mps' if torch.backends.mps.is_available() else 'cpu', without_mask = False):
        super(TrainerBERT, self).__init__(model, criterion, optimizer, device, without_mask)
        self.eval_device = self.device
    def train_step(self, x, y, mask=None):
        "Training step"
        self.model.train()
        self.model.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        if self.without_mask:
            outputs = self.model(x)
        else:
            outputs = self.model(x, attention_mask=mask)

        y_pred = outputs.logits
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        # compute loss
        self.train_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.train_acc += correct


    def val_step(self, x, y, mask=None):
        "Validation step"
        self.model.eval()
        self.model.to(self.eval_device)
        x, y = x.to(self.eval_device), y.to(self.eval_device)
        if mask is not None:
            mask = mask.to(self.eval_device)

        if self.without_mask:
            outputs = self.model(x)
        else:
            outputs = self.model(x,  attention_mask=mask)
        y_pred = outputs.logits
        loss = self.criterion(y_pred, y)
        # compute loss
        self.val_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.val_acc += correct