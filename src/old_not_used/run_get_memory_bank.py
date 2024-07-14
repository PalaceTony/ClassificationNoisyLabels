from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from PreResNet import *
import hydra
from omegaconf import DictConfig
import logging

from lib.utils import (
    MemoryBank,
    seed_everything,
    SemiLoss,
    NegEntropy,
)
from lib import dataloader as dataloader
from trainer_get_memory_bank import Trainer


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    logger = logging.getLogger("Wafer")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.seed)

    memory_bank_net1 = MemoryBank(q_length=cfg.q_length)
    memory_bank_net2 = MemoryBank(q_length=cfg.q_length)

    net1 = ResNet18(num_classes=cfg.num_class).to(device)
    net2 = ResNet18(num_classes=cfg.num_class).to(device)

    loader = dataloader.wafer_dataloader(
        cfg.dataset,
        r=cfg.r,
        noise_mode=cfg.noise_mode,
        batch_size=cfg.batch_size,
        num_workers=5,
        root_dir=cfg.data_path,
        noise_file="%s/%.1f_%s.json" % (cfg.data_path, cfg.r, cfg.noise_mode),
    )

    optimizer1 = optim.SGD(
        net1.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4
    )
    optimizer2 = optim.SGD(
        net2.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4
    )
    criterion = SemiLoss(cfg)
    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    if not cfg.robust_directly:
        Trainer(
            cfg,
            net1,
            net2,
            loader,
            optimizer1,
            optimizer2,
            memory_bank_net1,
            memory_bank_net2,
            criterion,
            CE,
            CEloss,
            conf_penalty,
            logger,
            device,
        ).train()


if __name__ == "__main__":
    main()
