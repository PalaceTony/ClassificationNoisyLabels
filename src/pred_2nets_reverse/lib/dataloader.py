from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle

    with open(file, "rb") as fo:
        dict = cPickle.load(fo, encoding="latin1")
    return dict


class wafer_dataset(Dataset):
    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        root_dir,
        transform,
        mode,
        noise_file="",
        pred=[],
        probability=[],
        logger=None,
    ):
        self.original_indices = []
        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.target_labels = [0, 1, 2, 3, 4, 6, 10, 14, 26]
        self.transition = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 6,
            6: 10,
            10: 14,
            14: 26,
            26: 0,
        }  # class transition for asymmetric noise within target labels

        if self.mode == "test":
            if dataset == "Wafer":
                self.test_data = np.load("data/Wafer/3splits/test_data.npz")[
                    "images"
                ].astype(np.uint8)
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = np.load("data/Wafer/3splits/test_data.npz")[
                    "labels"
                ].tolist()
        elif self.mode == "validation":
            if dataset == "Wafer":
                self.val_data = np.load("data/Wafer/3splits/val_data.npz")[
                    "images"
                ].astype(np.uint8)
                self.val_data = self.val_data.transpose((0, 2, 3, 1))
                self.val_label = np.load("data/Wafer/3splits/val_data.npz")[
                    "labels"
                ].tolist()
        else:
            train_data = []
            train_label = []
            if dataset == "Wafer":
                train_data = np.load("data/Wafer/3splits/train_data.npz")[
                    "images"
                ].astype(np.uint8)
                train_label = np.load("data/Wafer/3splits/train_data.npz")[
                    "labels"
                ].tolist()
            # train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                target_labels = [0, 1, 2, 3, 4, 6, 10, 14, 26]
                noise_label = []
                idx = [
                    i
                    for i in range(len(train_label))
                    if train_label[i] in target_labels
                ]
                random.shuffle(idx)
                num_noise = int(
                    self.r * len(idx)
                )  # Calculate the number of noises based on filtered indices
                noise_idx = idx[:num_noise]
                for i in range(len(train_label)):
                    if i in noise_idx:
                        noise_type = random.choice(["sym", "asym"])
                        if noise_type == "sym":
                            noiselabel = random.choice(target_labels)
                        elif noise_type == "asym":
                            noiselabel = self.transition[train_label[i]]
                        noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

            if self.mode == "all":
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = np.array(noise_label) == np.array(train_label)
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    logger.info(
                        "Numer of labeled samples:%d   AUC:%.3f" % (pred.sum(), auc)
                    )
                    self.original_indices = pred_idx  # Store the original indices

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                    self.original_indices = pred_idx  # Store the original indices

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                logger.info(
                    "%s data has a size of %d" % (self.mode, len(self.noise_label))
                )

    def __getitem__(self, index):
        if self.mode == "labeled":
            img, target, prob = (
                self.train_data[index],
                self.noise_label[index],
                self.probability[index],
            )
            original_index = self.original_indices[index]
            img = np.squeeze(img, axis=2)
            img = Image.fromarray(img, mode="L")
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob, original_index
        elif self.mode == "unlabeled":
            img = self.train_data[index]
            target = self.noise_label[index]
            prob = self.probability[index]
            original_index = self.original_indices[index]
            img = np.squeeze(img, axis=2)
            img = Image.fromarray(img, mode="L")
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob, original_index
        elif self.mode == "all":
            img, target = self.train_data[index], self.noise_label[index]
            img = np.squeeze(img, axis=2)
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
            return img, target, index
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = np.squeeze(img, axis=2)
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
            return img, target
        elif self.mode == "validation":
            img, target = self.val_data[index], self.val_label[index]
            img = np.squeeze(img, axis=2)
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test" and self.mode != "validation":
            return len(self.train_data)
        elif self.mode == "test":
            return len(self.test_data)
        elif self.mode == "validation":
            return len(self.val_data)


class wafer_dataloader:
    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        batch_size,
        num_workers,
        root_dir,
        noise_file="",
    ):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file

        if self.dataset == "Wafer":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(52, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.transform_test = transforms.Compose([transforms.ToTensor()])

    def run(self, mode, pred=[], prob=[], logger=None):
        if mode == "warmup":
            all_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="all",
                noise_file=self.noise_file,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader

        elif mode == "train":
            labeled_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="labeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                logger=logger,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="unlabeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                logger=logger,
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return labeled_trainloader, unlabeled_trainloader

        elif mode == "test":
            test_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="test",
                logger=logger,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
        elif mode == "validation":
            val_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="validation",
                logger=logger,
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader

        elif mode == "eval_train":
            eval_dataset = wafer_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="all",
                noise_file=self.noise_file,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader
