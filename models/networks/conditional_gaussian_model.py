import math

import torch
import torch.nn as nn

from ..ops import lower_bound, signum


class GaussianModel(nn.Module):
    def __init__(self, scale_bound, likelihood_bound, sign_reversal=True):
        super(GaussianModel, self).__init__()
        # Save the input parameters
        self.scale_bound = float(scale_bound)  # 0.1
        self.likelihood_bound = float(likelihood_bound)  # 1e-9
        self.sign_reverse = bool(sign_reversal)

    def forward(self, inputs, loc, scale):
        # Set the scale lower bound， max or torch.clamp is ok
        denominator = lower_bound(scale, self.scale_bound) * math.sqrt(2.0)

        # Compute the probabilities
        values = inputs - loc  # loc->mu
        lower = values - 0.5
        upper = values + 0.5
        if not self.sign_reverse:
            likelihood = 1 / 2 * (torch.erf(upper / denominator) - torch.erf(lower / denominator))
        else:
            sign = signum(values).detach()
            likelihood = 1 / 2 * torch.abs(
                torch.erf(-sign * upper / denominator) - torch.erf(-sign * lower / denominator)
            )
        likelihood = lower_bound(likelihood, self.likelihood_bound)

        decor_latents = (inputs - loc) / denominator

        return likelihood, decor_latents
