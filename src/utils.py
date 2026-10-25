import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_logger(args):
    log_filename = os.path.join(args.root_path, "log", f"{args.dataset}.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def intrusion_metrics(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    return {"ACC": round(accuracy_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4)}


def detection_metrics(y_true, y_pred, scores):
    y_true, y_pred, scores = y_true.cpu().numpy(), y_pred.cpu().numpy(), scores.cpu().numpy()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return {"AUC": round(roc_auc_score(y_true, scores), 4),
            "TPR": round(tp / (tp + fn), 4),
            "FPR": round(fp / (fp + tn), 4)}
