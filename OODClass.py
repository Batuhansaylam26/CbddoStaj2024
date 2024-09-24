import torch.nn as nn
import torch
import numpy as np


class OODClass:
    def __init__(self, temperature, model, data_loader, device, epsilon=0.01):
        self.temperature = temperature
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.data_loader = data_loader

    def temperature_scaling(self):
        return self.logits / self.temperature

    def compute_softmax(self, logits):
        return nn.functional.softmax(logits, dim=1)

    def apply_odin_perturbations(self, x):
        x.requires_grad = True
        self.logits = self.model(x)
        probs = self.compute_softmax(self.logits)

        true_class = torch.argmax(probs, dim=1, keepdim=True)
        loss = -torch.log(probs.gather(1, true_class).squeeze())

        # Reduce loss to scalar
        loss = loss.mean()  # Use mean to ensure scalar loss for gradient computation

        self.model.zero_grad()
        loss.backward()

        # Create perturbation
        perturbation = self.epsilon * x.grad.sign()
        self.perturbed_x = x + perturbation
        return self.perturbed_x

    def detect_ood(self):
        self.model.eval()
        softmax_scores = []
        ood_scores = []
        self.testPredictionsList = []
        for images, _ in self.data_loader:
            images = images.to(self.device)
            self.logits = self.model(images)
            probs = self.compute_softmax(self.logits)
            softmax_scores.append(probs.max(dim=1)[0].detach().cpu().numpy())
            testPredictions = torch.max(self.logits, 1)[1].to(self.device)
            self.testPredictionsList.append(testPredictions)
            perturbed_images = self.apply_odin_perturbations(images)
            perturbed_images = perturbed_images.to(self.device)
            perturbed_logits = self.model(perturbed_images)
            perturbed_probs = self.compute_softmax(perturbed_logits)
            ood_scores.append(perturbed_probs.max(dim=1)[0].detach().cpu().numpy())

        self.softmax_scores = np.concatenate(softmax_scores)
        self.ood_scores = np.concatenate(ood_scores)
        return self.softmax_scores, self.ood_scores
