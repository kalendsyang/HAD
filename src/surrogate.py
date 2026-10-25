import torch


@torch.no_grad()
def generate_surrogates(ids, x, y, mfs, sigma=0.5, epsilon=1.0, max_iter=100):
    """Replace a sigma fraction of each malicious sample's MFS with values
    from a random donor, keeping candidates within the L2 budget that the
    IDS classifies as benign."""
    mfs = torch.as_tensor(mfs, device=x.device)
    k = max(1, round(sigma * len(mfs)))
    source = x[y == 1]
    pending = torch.arange(len(source), device=x.device)
    surrogates = []

    for _ in range(max_iter):
        if len(pending) == 0:
            break
        origin = source[pending]
        donor = x[torch.randint(len(x), (len(pending),), device=x.device)]
        cols = mfs[torch.rand(len(pending), len(mfs), device=x.device).argsort(1)[:, :k]]
        candidate = origin.scatter(1, cols, donor.gather(1, cols))

        dist = torch.norm(candidate - origin, dim=1)
        valid = (dist > 0) & (dist <= epsilon) & (ids.predict(candidate) == 0)
        surrogates.append(candidate[valid])
        pending = pending[~valid]

    return torch.cat(surrogates)
