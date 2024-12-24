import torch
from callback import callback

callback = callback()


class Train:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []
        self.val_loss_list = []
        self.val_accuracy_list = []
        self.predictions_list = []
        self.valPredictions_list = []

    def train(self):
        count = 0
        train_loss = 0
        total = 0
        correct = 0

        for batchID, (images, labels) in enumerate(self.train_loader):
            self.batchID = batchID
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()
            train_loss += self.loss.item()
            predictions = torch.max(outputs, 1)[1].to(self.device)
            self.predictions_list.append(predictions)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            callback.on_batch_end(
                self.batchID,
                logs={
                    "loss": self.loss / (self.batchID + 1),
                    "accuracy": correct * 100 / total,
                },
            )
        self.actualLoss = train_loss / len(self.train_loader)
        self.accuracy = correct * 100 / total
        self.loss_list.append(self.loss.data)
        self.iteration_list.append(count)
        self.accuracy_list.append(self.accuracy)

    def validate(self):
        valTotal = 0
        valCorrect = 0
        valLoss = 0
        for images, labels in self.val_loader:
            self.model.eval()
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            val_loss = self.criterion(outputs, labels)
            valLoss += val_loss.item()
            valPredictions = torch.max(outputs, 1)[1].to(self.device)
            self.valPredictions_list.append(valPredictions)
            valCorrect += (valPredictions == labels).sum().item()

            valTotal += len(labels)
        self.actualValLoss = valLoss / len(self.val_loader)
        self.valAccuracy = valCorrect * 100 / valTotal
        self.val_loss_list.append(valLoss)
        self.val_accuracy_list.append(self.valAccuracy)
