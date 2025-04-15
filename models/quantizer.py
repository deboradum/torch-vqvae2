import torch

import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, k, d, beta):
        super().__init__()

        self.k = k
        self.d = d
        self.beta = beta
        self.e = nn.Embedding(k, d)

        init_fn = nn.init.uniform(low=-1.0 / self.k, high=1.0 / self.k)
        self.e.apply(init_fn)

    def __call__(self, z):
        # (b, c, w, h) > (b, w, h, c) & flatten
        # z = z.transpose(0, 2, 3, 1)
        z = z.contiguous()
        z_flat = z.view(-1, self.d)


        # || z_{e}(x) - e_{j} ||_{2}
        dists = (
            (z_flat**2).sum(dim=1, keepdim=True)
            + (self.e.weight**2).sum(dim=1, keepdim=True).T
            - 2 * torch.matmul(z_flat, self.e.weight.T)
        )

        # The posterior categorical distribution q(z|x) probabilities are defined as one-hot as follows:
        # q(z=k|x) = { 1 for k = argmin_{j} || z_{e}(x) - e_{j} ||_{2}
        #            { 0 else
        closest_indices = dists.argmin(dim=1)
        one_hot = F.one_hot(closest_indices, num_classes=self.k).float()
        z_q = self.e(closest_indices).reshape(*z.shape)

        codebook_loss = ((z_q.detach() - z) ** 2).mean()
        commitment_loss = self.beta * ((z_q - z.detach()) ** 2).mean()
        loss = codebook_loss + commitment_loss

        e_mean = one_hot.mean(0)
        eps = 1e-10
        perplexity = (-(e_mean + eps) * (e_mean + eps).log()).sum().exp()

        z_q = z + (z_q - z).detach()
        # z_q = mx.contiguous(z_q.transpose(0, 3, 1, 2))
        z_q = z_q.contiguous()

        return loss, z_q, perplexity
