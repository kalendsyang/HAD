import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


class DNN(nn.Module):
    """Backbone IDS classifier."""

    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(1)


class Baseline:
    """Classical IDS models (RF, GNB) with the same tensor interface as DNN."""

    def __init__(self, name, seed=2024):
        if name == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        else:
            self.model = GaussianNB()

    def fit(self, x, y):
        self.model.fit(x.cpu().numpy(), y.cpu().numpy())
        return self

    def predict(self, x):
        return torch.as_tensor(self.model.predict(x.cpu().numpy()), dtype=torch.long, device=x.device)


class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class HAD(nn.Module):
    """Dual encoders over the MFS/NFS views; the dispersion score is the
    distance between the two embeddings."""

    def __init__(self, mfs, nfs, emb_dim=16):
        super().__init__()
        self.register_buffer("mfs", torch.as_tensor(mfs, dtype=torch.long))
        self.register_buffer("nfs", torch.as_tensor(nfs, dtype=torch.long))
        self.register_buffer("tau", torch.tensor(float("inf")))
        self.mfs_encoder = Encoder(len(mfs), emb_dim)
        self.nfs_encoder = Encoder(len(nfs), emb_dim)

    def forward(self, x):
        z_m = self.mfs_encoder(x[:, self.mfs])
        z_n = self.nfs_encoder(x[:, self.nfs])
        return torch.norm(z_m - z_n, dim=1)

    @torch.no_grad()
    def calibrate(self, x_benign, fpr=0.05):
        self.tau = torch.quantile(self.forward(x_benign), 1 - fpr)

    @torch.no_grad()
    def predict(self, x, ids=None):
        pred = (self.forward(x) > self.tau).long()
        if ids is not None:
            pred = pred | ids.predict(x)
        return pred
