import torch
from torch.nn import functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, class_idx):
        print(self.gradients.shape)
        pooled_gradients = torch.mean(self.gradients, dim=[0])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap).cpu().detach().numpy()

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        return heatmap

    def __call__(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        print('-' * 100)
        print(output)
        class_score = output[0][0][class_idx]
        class_score.backward(retain_graph=True)
        heatmap = self.generate_heatmap(class_idx)
        return heatmap
