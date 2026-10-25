import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from src.attack import pgd_attack
from src.data import get_data
from src.feature import shap_values, evasion_potential, partition_features
from src.loss import HierarchicalLoss, SURROGATE
from src.model import DNN, Baseline, HAD
from src.surrogate import generate_surrogates
from src.utils import fix_seed, get_logger, intrusion_metrics, detection_metrics

warnings.filterwarnings("ignore")


def train_ids(args, logger, x, y):
    """Train the backbone IDS."""

    model = DNN(x.shape[1]).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for idx in torch.randperm(len(x), device=x.device).split(args.batch_size):
            loss = criterion(model(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        logger.info(f'IDS Epoch: [{epoch}/{args.epochs}], loss={epoch_loss / len(x):.6f}')

    return model.eval()


def feature_partition(args, logger, ids, x, y):
    """Stage 1: partition features into MFS and NFS by evasion potential."""

    background = x[torch.randperm(len(x))[:args.shap_background]]
    idx = torch.randperm(len(x))[:args.shap_samples]
    values = shap_values(ids, background, x[idx])
    ep = evasion_potential(values, y[idx].cpu().numpy())
    mfs, nfs = partition_features(ep, args.rho)
    logger.info(f'MFS: {mfs.tolist()}')
    logger.info(f'NFS: {nfs.tolist()}')
    return mfs, nfs


def train_had(args, logger, had, x, y):
    """Stage 3: hierarchical contrastive training of the dual encoders."""

    optimizer = Adam(had.parameters(), lr=args.lr)
    criterion = HierarchicalLoss(args.delta, args.lambda_b, args.lambda_s, args.lambda_m)
    groups = [torch.where(y == c)[0] for c in torch.unique(y)]
    per_group = args.batch_size // len(groups)
    steps = len(x) // args.batch_size

    for epoch in range(args.epochs):
        had.train()
        epoch_loss = {"L": 0.0, "L_b": 0.0, "L_s": 0.0, "L_m": 0.0}
        for _ in range(steps):
            idx = torch.cat([g[torch.randint(len(g), (per_group,), device=g.device)] for g in groups])
            loss, terms = criterion(had(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss["L"] += loss.item()
            for k, v in terms.items():
                epoch_loss[k] += v
        logger.info(f'HAD Epoch: [{epoch}/{args.epochs}], '
                    + ', '.join(f'{k}={v / steps:.6f}' for k, v in epoch_loss.items()))

    return had.eval()


@torch.no_grad()
def evaluate(args, logger, had, models, x_test, y_test, x_adv):
    """Adversarial sample detection and intrusion detection under attack."""

    ids = models["DNN"]
    attacked = (y_test == 1) & (ids.predict(x_test) == 1)
    success = attacked & (ids.predict(x_adv) == 0)
    logger.info(f'Attack success: {success.sum().item()}/{attacked.sum().item()}')

    # Successful adversarial samples mixed with benign test samples.
    x_det = torch.cat([x_test[y_test == 0], x_adv[success]])
    y_det = torch.cat([torch.zeros((y_test == 0).sum(), dtype=torch.long, device=args.device),
                       torch.ones(success.sum(), dtype=torch.long, device=args.device)])
    logger.info(f'Adversarial detection (HAD): {detection_metrics(y_det, had.predict(x_det), had(x_det))}')

    # Test set in which attacked malicious samples are replaced by adversarial ones.
    for name, model in models.items():
        logger.info(f'{name}: clean {intrusion_metrics(y_test, model.predict(x_test))}, '
                    f'adversarial {intrusion_metrics(y_test, model.predict(x_adv))}, '
                    f'adversarial+HAD {intrusion_metrics(y_test, had.predict(x_adv, model))}')
    logger.info(f'HAD standalone: clean {intrusion_metrics(y_test, had.predict(x_test))}, '
                f'adversarial {intrusion_metrics(y_test, had.predict(x_adv))}')


def main(args, logger):
    x_train, y_train, x_test, y_test = get_data(args.root_path, args.dataset, args.seed)
    x_fit, x_cal, y_fit, y_cal = train_test_split(x_train, y_train, test_size=args.cal_rate,
                                                  stratify=y_train, random_state=args.seed)
    to_tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=args.device)
    x_fit, x_cal, x_test = to_tensor(x_fit), to_tensor(x_cal), to_tensor(x_test)
    y_fit, y_cal, y_test = to_tensor(y_fit, torch.long), to_tensor(y_cal, torch.long), to_tensor(y_test, torch.long)

    ids = train_ids(args, logger, x_fit, y_fit)
    models = {"DNN": ids,
              "RF": Baseline("rf", args.seed).fit(x_fit, y_fit),
              "GNB": Baseline("gnb").fit(x_fit, y_fit)}

    mfs, nfs = feature_partition(args, logger, ids, x_fit, y_fit)

    x_sur = generate_surrogates(ids, x_fit, y_fit, mfs, args.sigma, args.epsilon, args.max_iter)
    logger.info(f'Generated {len(x_sur)} surrogate samples')

    had = HAD(mfs, nfs, args.emb_dim).to(args.device)
    x_had = torch.cat([x_fit, x_sur])
    y_had = torch.cat([y_fit, torch.full((len(x_sur),), SURROGATE, device=args.device)])
    train_had(args, logger, had, x_had, y_had)
    had.calibrate(x_cal[y_cal == 0], args.fpr)
    logger.info(f'Threshold tau={had.tau.item():.6f}')

    if args.adv_path:
        x_adv = to_tensor(np.load(args.adv_path))
    else:
        x_adv = pgd_attack(ids, x_test, y_test, args.eps, args.alpha, args.steps)
    evaluate(args, logger, had, models, x_test, y_test, x_adv)

    os.makedirs(os.path.join(args.root_path, "model"), exist_ok=True)
    torch.save({"ids": ids.state_dict(), "had": had.state_dict()},
               os.path.join(args.root_path, "model", f"{args.dataset}.pth"))


def argument_parser():
    parser = argparse.ArgumentParser(description='HAD')
    parser.add_argument('--dataset', type=str, default='nsl', choices=['nsl', 'cic'], help='Dataset name')
    parser.add_argument('--root_path', type=str, default='./', help='Root path')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')

    parser.add_argument('--rho', type=float, default=0.5, help='Fraction of features assigned to MFS')
    parser.add_argument('--shap_samples', type=int, default=1000, help='Samples explained by SHAP')
    parser.add_argument('--shap_background', type=int, default=100, help='SHAP background size')
    parser.add_argument('--sigma', type=float, default=0.5, help='Fraction of MFS replaced in surrogates')
    parser.add_argument('--epsilon', type=float, default=1.0, help='L2 budget for surrogates')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum surrogate attempts')

    parser.add_argument('--emb_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--delta', type=float, default=1.0, help='Margin of the hierarchical loss')
    parser.add_argument('--lambda_b', type=float, default=1.0, help='Weight of L_b')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='Weight of L_s')
    parser.add_argument('--lambda_m', type=float, default=1.0, help='Weight of L_m')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--cal_rate', type=float, default=0.2, help='Held-out rate for threshold calibration')
    parser.add_argument('--fpr', type=float, default=0.05, help='Target false positive rate')

    parser.add_argument('--eps', type=float, default=0.1, help='PGD perturbation budget')
    parser.add_argument('--alpha', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--steps', type=int, default=40, help='PGD iterations')
    parser.add_argument('--adv_path', type=str, default=None, help='Adversarial test set (.npy) from another attack')
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    args.device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')
    fix_seed(args.seed)
    logger = get_logger(args)
    logger.info('======START======')
    logger.info('Args:' + str(args))
    main(args, logger)
    logger.info('======END=======')
