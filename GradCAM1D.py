import torch
import torch.nn.functional as F


class GradCAM1D:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient = None
        self.activation = None
        self.model.eval()
        self.register_hooks()

    def save_gradients(self, grad):
        self.gradients = grad

    def save_activations(self, module, input, output):
        self.activations = output

    def register_hooks(self):
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(self.save_activations)
        layer.register_backward_hook(lambda module, grad_in, grad_out: self.save_gradients(grad_out[0]))

    def get_cam_weights(self, grads, activations):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        sum_activations = activations.sum(dim=2, keepdim=True)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij
        weights = torch.sum(weights, dim=2)
        return weights

    def generate_cam(self, input_spectra, target_class):
        self.model.eval()
        output = self.model(input_spectra)[0]  # Assuming the first output is for classification task
        class_score = output[:, target_class]

        self.model.zero_grad()
        class_score.backward()

        # pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        # for i in range(self.activations.shape[1]):
        #     self.activations[:, i, :] *= pooled_gradients[i]

        weights = self.get_cam_weights(self.gradients, self.activations)
        weights = torch.squeeze(weights)
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :] *= weights[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap
