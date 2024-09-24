import torch
import torch.nn as nn
from dataPrep import Data
from torch.utils.data import DataLoader
from model import OODCNN
from train import Train
from callback import callback, EarlyStopping
from OODClass import OODClass
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
data = Data()
train = data.returnTrain()
test = data.returnTest()
val = data.returnVal()
train_loader = DataLoader(
    train,
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(test, batch_size=64, shuffle=False)
val_loader = DataLoader(val, batch_size=64, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_logs = []


model = OODCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
earlyStoppingCallback = EarlyStopping()
callback = callback()
trainSteps = Train(
    model=model,
    device=device,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
)
epochs = 100

for epoch in range(epochs):
    model.train()
    callback.on_epoch_begin(epoch)
    trainSteps.train()
    trainSteps.validate()
    logs = {
        "loss": trainSteps.actualLoss,
        "accuracy": trainSteps.accuracy,
        "valLoss": trainSteps.actualValLoss,
        "valAccuracy": trainSteps.valAccuracy,
    }
    callback.on_epoch_end(epoch, logs)

    if earlyStoppingCallback and earlyStoppingCallback.on_epoch_end(
        epoch, logs, val=True
    ):
        break
    else:
        torch.save(model.state_dict(), "model.pth")
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])
model = OODCNN().to(device)
model.load_state_dict(torch.load("model.pth"))

temperature = 1000.0
epsilon = 0.01
ood = OODClass(
    temperature=temperature,
    epsilon=epsilon,
    model=model,
    data_loader=test_loader,
    device=device,
)

softmax_scores, ood_scores = ood.detect_ood()
is_fmnist = [False] * 5000 + [True] * 1000
threshold = 0.9
is_ood = ood_scores < threshold
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc


tn, fp, fn, tp = confusion_matrix(is_fmnist, is_ood).ravel()


from sklearn.metrics import roc_curve

# FPR@95 hesaplama
fpr, tpr, thresholds = roc_curve(is_fmnist, ood_scores)

# TPR = 0.95'e en yakın değeri bul
fpr_at_95 = fpr[np.argmax(tpr >= 0.95)]
print(f"FPR@95: {fpr_at_95}")



auroc = roc_auc_score(is_fmnist, ood_scores)


precision, recall, _ = precision_recall_curve(is_fmnist, ood_scores)
aupr = auc(recall, precision)

import seaborn as sns
import matplotlib.pyplot as plt
cm = np.array([[tn, fp],
               [fn, tp]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted ID', 'Predicted OOD'], yticklabels=['Actual ID', 'Actual OOD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(f"Confusion Matrix: \nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
print(f"FPR: {fpr}")
print(f"AUROC: {auroc}")
print(f"AUPR: {aupr}")

print(len(data.returnTest()))
logging.info(f"Detected OOD samples: {np.sum(is_ood)} out of {len(is_ood)}")
