import torch
import torch.nn.functional as F


def pgd_attack(model, x, y, eps=0.1, alpha=0.01, steps=40):
    """L-inf PGD against the IDS on malicious samples it classifies correctly.

    Returns a copy of x in which the attacked samples are replaced by their
    adversarial versions.
    """
    target = (y == 1) & (model.predict(x) == 1)
    origin = x[target]
    label = torch.ones(len(origin), dtype=torch.long, device=x.device)

    adv = (origin + torch.empty_like(origin).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), label)
        grad, = torch.autograd.grad(loss, adv)
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, origin - eps), origin + eps).clamp(0, 1)

    x_adv = x.clone()
    x_adv[target] = adv
    return x_adv
