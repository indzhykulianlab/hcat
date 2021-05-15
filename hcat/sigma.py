import torch


class Sigma:
    def __init__(self, adjustments, initial_sigma = [0.1, 0.1, 0.8], device='cpu'):
        self.adjutments = adjustments
        self.device = device
        if isinstance(initial_sigma, list):
            self.initial_sigma = torch.tensor(initial_sigma, device=device)
        else:
            self.initial_sigma = initial_sigma

        values = [1]  # Initial values set so sigma isnt zero at 3 zero
        epochs = [-1]
        for d in adjustments:
            values.append(d['multiplier'])
            epochs.append(d['epoch'])

        self.values = torch.tensor(values, device=self.device)
        self.epochs = torch.tensor(epochs, device=self.device)

    def __call__(self, e):
        multiplier = self.values[self.epochs < e].prod()

        return self.initial_sigma * multiplier
