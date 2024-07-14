import collections
import numpy as np
import torch
import random
import logging
import pickle
from torch.nn import functional as F
import os


class MemoryBank:
    def __init__(self, q_length, threshold=0.6, total_entries=26610):
        self.q_length = q_length
        self.threshold = threshold
        self.memory = {
            i: collections.deque(maxlen=q_length) for i in range(total_entries)
        }
        self.noisy_labels = {}  # Dictionary to store noisy labels

    def update_memory(self, indices, predictions):
        for idx, pred in zip(indices, predictions):
            self.memory[idx].append(pred)

    def store_noisy_labels(self, indices, labels):
        for idx, label in zip(indices, labels):
            self.noisy_labels[idx] = label

    def get_consistent_samples(self, indices, labels):
        consistent_indices = []
        consistent_labels = []
        for idx, label in zip(indices, labels):
            preds = list(self.memory[idx])
            if len(preds) < self.q_length:
                consistent_indices.append(idx)
                consistent_labels.append(label)
            else:
                agreement_ratio = np.mean(np.array(preds) == label)
                if agreement_ratio >= self.threshold:
                    consistent_indices.append(idx)
                    consistent_labels.append(label)
        return consistent_indices, consistent_labels


def save_memory_banks(cfg, memory_bank_net1, memory_bank_net2):
    with open(os.path.join(cfg.log_dir, "memory_bank_net1.pkl"), "wb") as f:
        pickle.dump(memory_bank_net1, f)
    with open(os.path.join(cfg.log_dir, "memory_bank_net2.pkl"), "wb") as f:
        pickle.dump(memory_bank_net2, f)


def load_memory_banks(cfg):
    with open(os.path.join(cfg.best_model_path, "memory_bank_net1.pkl"), "rb") as f:
        memory_bank_net1 = pickle.load(f)
    with open(os.path.join(cfg.best_model_path, "memory_bank_net2.pkl"), "rb") as f:
        memory_bank_net2 = pickle.load(f)
    return memory_bank_net1, memory_bank_net2


def seed_everything(seed=42):
    # Seed all
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def linear_rampup(cfg, current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return cfg.lambda_u * float(current)


class SemiLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(self.cfg, epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))
