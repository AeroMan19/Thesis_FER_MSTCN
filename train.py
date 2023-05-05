import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from models import *
from collections import Counter

class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def find_majority(self, x):
        val, _ = torch.mode(x, dim=1)
        return val

    def train(self, save_dir, dataset, val_dataset, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_acc = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            pred_l = []
            gt_l = []

            print('Begining Epoch %d / %d' % (epoch, num_epochs))
            # ------------------training------------------
            for feat, gt, fname in dataset:
                feat, gt = feat.to(device), gt.to(device)
                optimizer.zero_grad()
                feat = torch.swapaxes(feat, 1, 2)
                predictions = self.model(feat)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)

                #print('before:', gt.shape, predicted.shape)
                predicted = self.find_majority(predicted)
                gt = self.find_majority(gt[:, :, 0])
                #print('after:', gt.shape, predicted.shape)
                correct += ((predicted == gt).float()).sum().item()
                total += feat.size(0)

                predicted = torch.squeeze(predicted, dim=0).cpu().detach().numpy()
                gt = torch.squeeze(gt, dim=0).cpu().detach().numpy()

                pred_l.append(predicted)
                gt_l.append(gt)

            ccc = self.eval_ccc(np.asarray(pred_l), np.asarray(gt_l))

            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct) / total))

            print("[epoch %d]: Trainnig epoch loss = %f,   acc = %f,  ccc = %f" % (epoch + 1, epoch_loss / len(dataset),
                                                               float(correct) / total, float(ccc) / total))

            # ------------------validation------------------

            val_epoch_loss = 0
            val_correct = 0
            val_total = 0

            pred_l = []
            gt_l = []

            for feat, gt, fname in val_dataset:
                feat, gt = feat.to(device), gt.to(device)
                optimizer.zero_grad()
                feat = torch.swapaxes(feat, 1, 2)
                predictions = self.model(feat)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16))

                val_epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = self.find_majority(predicted)
                gt = self.find_majority(gt[:, :, 0])
                val_correct += ((predicted == gt).float()).sum().item()
                val_total += feat.size(0)

                predicted = torch.squeeze(predicted, dim=0).cpu().detach().numpy()
                gt = torch.squeeze(gt, dim=0).cpu().detach().numpy()

                pred_l.append(predicted)
                gt_l.append(gt)

            val_ccc = self.eval_ccc(np.asarray(pred_l), np.asarray(gt_l))

            print("[epoch %d]: Validation epoch loss = %f,   acc = %f,  ccc = %f" % (epoch + 1, val_epoch_loss / len(val_dataset),
                                                               float(val_correct) / val_total, float(val_ccc) / total))

            if (float(val_correct) / val_total) > best_val_acc:
                torch.save(self.model.state_dict(), save_dir + "/acc_best.model")
                torch.save(optimizer.state_dict(), save_dir + "/acc_best.opt")
                best_val_acc = float(val_correct) / val_total 

                with open(save_dir + '/training_log.txt', 'a') as fd:
                    fd.write('\n-----------------------------------------epoch: {}/{}-----------------------------------------'.format(epoch, num_epochs))
                    fd.write("\n [epoch %d]: Training   epoch loss = %f,   acc = %f,  ccc = %f" % (epoch + 1, epoch_loss / len(dataset), float(correct) / total, float(ccc) / total))
                    fd.write("\n [epoch %d]: Validation epoch loss = %f,   acc = %f,  ccc = %f" % (epoch + 1, val_epoch_loss / len(val_dataset), float(val_correct) / val_total, float(val_ccc) / total))

                print('best model saved....')

    def eval_ccc(self, y_true, y_pred):
        """Computes concordance correlation coefficient."""

        print(y_true, y_pred)
        true_mean = np.mean(y_true)
        true_var = np.var(y_true)
        pred_mean = np.mean(y_pred)
        pred_var = np.var(y_pred)
        covar = np.cov(y_true, y_pred, bias=True)[0][1]
        ccc = 2 * covar / (true_var + pred_var + (pred_mean - true_mean) ** 2)
        return ccc

    # def predict(): # make predictions on the test data
