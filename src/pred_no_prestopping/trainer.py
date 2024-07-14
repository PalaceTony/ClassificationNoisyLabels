import torch
from sklearn.mixture import GaussianMixture
import os
import numpy as np
import pickle

from pred_no_prestopping.lib.utils import save_memory_banks, load_memory_banks
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Trainer(object):
    def __init__(
        self,
        cfg,
        net1,
        net2,
        loader,
        optimizer1,
        optimizer2,
        memory_bank_net1,
        memory_bank_net2,
        memory_bank_net3,
        memory_bank_net4,
        criterion,
        CE,
        CEloss,
        conf_penalty,
        logger,
        device,
        early_stop_epoch=0,
    ):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.net1 = net1
        self.net2 = net2
        self.loader = loader
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.memory_bank_net1 = memory_bank_net1
        self.memory_bank_net2 = memory_bank_net2
        self.memory_bank_net3 = memory_bank_net3
        self.memory_bank_net4 = memory_bank_net4
        self.criterion = criterion
        self.CE = CE
        self.CEloss = CEloss
        self.conf_penalty = conf_penalty
        self.logger = logger
        self.device = device
        self.early_stop_epoch = early_stop_epoch
        self.loader = loader
        self.warmup_trainloader = loader.run("warmup")
        self.test_loader = loader.run("test")
        self.val_loader = loader.run("validation")
        self.eval_loader = loader.run("eval_train")
        self.best_path = os.path.join(
            self.cfg.best_model_path, self.cfg.best_model_name
        )

    def warmup(self, epoch, net, optimizer, dataloader):
        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            if batch_idx >= self.cfg.warm_up_iter:
                break
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = self.CEloss(outputs, labels)
            penalty = self.conf_penalty(outputs)
            L = loss + penalty
            L.backward()
            optimizer.step()

            self.logger.info(
                "Noise %.1f Epoch [%3d/%3d] Iter[%3d/%3d] CE-loss: %.4f"
                % (
                    self.cfg.r,
                    epoch + 1,
                    self.cfg.num_epochs,
                    batch_idx + 1,
                    num_iter,
                    loss.item(),
                )
            )

    def train_epoch(
        self,
        epoch,
        net,
        net2,
        optimizer,
        labeled_trainloader,
        unlabeled_trainloader,
        trainNet1=False,
        secondTraining=False,
    ):
        net.train()
        net2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // self.cfg.batch_size) + 1
        total_masked_samples = 0  # Initialize counter for masked samples

        for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, ids) in enumerate(
            labeled_trainloader
        ):
            if batch_idx >= self.cfg.train_epoch_iter:
                break
            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()

            batch_size = inputs_x.size(0)
            labels_x = torch.zeros(batch_size, self.cfg.num_class).scatter_(
                1, labels_x.view(-1, 1), 1
            )

            w_x = w_x.view(-1, 1).type(torch.FloatTensor)
            inputs_x, inputs_x2, labels_x, w_x = (
                inputs_x.cuda(),
                inputs_x2.cuda(),
                labels_x.cuda(),
                w_x.cuda(),
            )
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

            with torch.no_grad():
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)

                pu = (
                    torch.softmax(outputs_u11, dim=1)
                    + torch.softmax(outputs_u12, dim=1)
                    + torch.softmax(outputs_u21, dim=1)
                    + torch.softmax(outputs_u22, dim=1)
                ) / 4
                ptu = pu ** (1 / self.cfg.T)

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)

                px = (
                    torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)
                ) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / self.cfg.T)

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            l = np.random.beta(self.cfg.alpha, self.cfg.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[: batch_size * 2]
            logits_u = logits[batch_size * 2 :]

            Lx, Lu, lamb = self.criterion(
                logits_x,
                mixed_target[: batch_size * 2],
                logits_u,
                mixed_target[batch_size * 2 :],
                epoch + batch_idx / num_iter,
                self.cfg.warm_up,
            )

            prior = torch.ones(self.cfg.num_class) / self.cfg.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_net = "Net1" if trainNet1 else "Net2"
            self.logger.info(
                "%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d] Labeled loss: %.2f  Unlabeled loss: %.2f"
                % (
                    training_net,
                    self.cfg.r,
                    epoch,
                    self.cfg.num_epochs,
                    batch_idx + 1,
                    num_iter,
                    Lx.item(),
                    Lu.item(),
                )
            )

            if secondTraining:
                self.logger.info(
                    f"Total masked samples in current batch: {total_masked_samples}"
                )

    def val(self, epoch, net1, net2):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        acc = 100.0 * correct / total
        self.logger.info(f"Val Epoch #{epoch}\t Accuracy: {acc:.2f}%\n")

        return acc

    def val_train(self, epoch, net1, net2):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, ids) in enumerate(self.eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                self.memory_bank_net3.update_memory(
                    ids.cpu().numpy(), predicted.cpu().numpy()
                )
            with open(
                os.path.join(self.cfg.log_dir, f"memory_bank_net3_epoch_{epoch}.pkl"),
                "wb",
            ) as f:
                pickle.dump(self.memory_bank_net3, f)

    def train(self):

        if self.cfg.load_model_during_first_train:
            checkpoint = torch.load(self.best_path)
            self.net1.load_state_dict(checkpoint["net1"])
            self.net2.load_state_dict(checkpoint["net2"])
            start_epoch = checkpoint["epoch"]
            self.logger.info("Models loaded")

            # self.memory_bank_net1, self.memory_bank_net2 = load_memory_banks(self.cfg)
            self.logger.info("Memory banks restarted")

        all_loss = [[], []]  # save the history of losses from two networks
        best_acc = 0
        not_improved_count = 0

        for epoch in (
            range(self.cfg.num_epochs + 1)
            if not self.cfg.load_model_during_first_train
            else range(36, 46)
        ):
            lr = self.cfg.lr
            if epoch >= 150:
                lr /= 10
            for param_group in self.optimizer1.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer2.param_groups:
                param_group["lr"] = lr

            if epoch < self.cfg.warm_up:
                self.logger.info("Warmup Net1")
                self.warmup(epoch, self.net1, self.optimizer1, self.warmup_trainloader)
                self.logger.info("Warmup Net2")
                self.warmup(epoch, self.net2, self.optimizer2, self.warmup_trainloader)
            else:
                prob1, all_loss[0] = self.eval_train(self.net1, all_loss[0])
                prob2, all_loss[1] = self.eval_train(self.net2, all_loss[1])
                pred1 = prob1 > self.cfg.p_threshold
                pred2 = prob2 > self.cfg.p_threshold

                self.logger.info("Train Net1")
                labeled_trainloader, unlabeled_trainloader = self.loader.run(
                    "train", pred2, prob2, logger=self.logger
                )  # co-divide
                self.train_epoch(
                    epoch,
                    self.net1,
                    self.net2,
                    self.optimizer1,
                    labeled_trainloader,
                    unlabeled_trainloader,
                    trainNet1=True,
                )  # train net1

                self.logger.info("Train Net2")
                labeled_trainloader, unlabeled_trainloader = self.loader.run(
                    "train", pred1, prob1, logger=self.logger
                )  # co-divide
                self.train_epoch(
                    epoch,
                    self.net2,
                    self.net1,
                    self.optimizer2,
                    labeled_trainloader,
                    unlabeled_trainloader,
                )  # train net2

                acc = self.val(epoch, self.net1, self.net2)

                if acc > best_acc:
                    not_improved_count = 0
                    best_acc = acc
                    self.early_stop_epoch = epoch

                    state = {
                        "net1": self.net1.state_dict(),
                        "net2": self.net2.state_dict(),
                        "acc": best_acc,
                        "epoch": self.early_stop_epoch,
                    }
                    torch.save(
                        state, os.path.join(self.cfg.log_dir, self.cfg.best_model_name)
                    )
                    self.logger.info(
                        "Saving current best model to "
                        + os.path.join(self.cfg.log_dir, self.cfg.best_model_name)
                        + "\n"
                    )
                else:
                    state = {
                        "net1": self.net1.state_dict(),
                        "net2": self.net2.state_dict(),
                        "acc": best_acc,
                        "epoch": self.early_stop_epoch,
                    }

                    # save unique for each epoch

                    torch.save(
                        state,
                        os.path.join(self.cfg.log_dir, f"model_epoch_{epoch}.pth"),
                    )

                    self.logger.info(
                        "Saving current Epoch model to " + self.cfg.log_dir + "\n"
                    )

                    not_improved_count += 1

            self.test(epoch, self.net1, self.net2)

    def eval_train(self, model, all_loss):
        model.eval()
        losses = torch.zeros(26610)
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(self.eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = self.CE(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        all_loss.append(losses)

        if (
            self.cfg.r == 0.9
        ):  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob, all_loss

    def test(self, epoch, net1, net2):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        acc = 100.0 * correct / total
        precision = precision_score(all_targets, all_predicted, average="macro")
        recall = recall_score(all_targets, all_predicted, average="macro")
        f1 = f1_score(all_targets, all_predicted, average="macro")

        self.logger.info(
            f"Test Epoch #{epoch}\t Accuracy: {acc:.2f}%\t Precision: {precision:.2f}\t Recall: {recall:.2f}\t F1-Score: {f1:.2f} \n"
        )
