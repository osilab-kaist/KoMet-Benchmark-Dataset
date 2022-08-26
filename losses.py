import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['CrossEntropyLoss', 'DiceLoss']


class ClassificationStat(nn.Module):
    """
    Superclass that provides functionality for evaluation based on original station observation information.
    """
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.reference = args.reference

    def get_stat(self, preds, targets, mode):
        """
        Get predictions and labels for original station observations.
        """
        if mode == 'train':
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
            b = pred_labels.shape[0]
            if b == 0:
                return

            pred_labels = pred_labels.squeeze(1).detach().reshape(b, -1)
            target_labels = targets.data.detach().reshape(b, -1)

        elif (mode == 'valid') or (mode == 'test'):
            # Old
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)

            # Current
            # preds = F.softmax(preds, dim=1)
            # true_probs = preds[:, 1, :].unsqueeze(1)
            # pred_labels = torch.where(true_probs > 0.05,
            #                           torch.ones(true_probs.shape).to(0),
            #                           torch.zeros(true_probs.shape).to(0))
            b, _, num_stn = pred_labels.shape
            assert (b, num_stn) == targets.shape

        pred_labels = pred_labels.squeeze(1).detach()
        target_labels = targets.data.detach()

        return pred_labels, target_labels

    def remove_missing_station(self, targets):
        _targets = targets.squeeze(0)
        targets_idx = (_targets >= 0).nonzero().cpu().tolist()  # [(x, y)'s] - hardcode
        return np.array(targets_idx)


class CrossEntropyLoss(ClassificationStat):
    def __init__(self, args, device, num_classes, experiment_name=None):
        super().__init__(args=args, num_classes=num_classes)
        self.device = device
        self.dataset_dir = args.dataset_dir
        self.experiment_name = experiment_name
        self.model_name = args.model
        self.args = args

    def forward(self, preds, targets, target_time, mode):
        """
        :param preds: model predictions in B x C x W x H format (batch size, channels, width, height)
        :param targets: targets in B x W x H format
        :param target_time:
        :param mode:
        :return:
        """
        if self.model_name == 'unet' or 'metnet':
            assert preds.shape[0] == targets.shape[0] and preds.shape[2] == targets.shape[1] and preds.shape[3] == \
                   targets.shape[2]
        elif self.model_name == 'convlstm':
            pass  # Chek the output size of convlstm model

        targets_shape = targets.shape

        if self.args.no_rain_ratio is not None:
            if self.args.target_precipitation == 'rain':  # Rain Case
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                rain_cnt = 0

                for rain_index in rain_counts_dict.keys():
                    if rain_index not in [0, -9999]:
                        rain_cnt += rain_counts_dict[rain_index]

                no_rain_cnt = int(rain_cnt * self.args.no_rain_ratio)

                if no_rain_cnt == 0:
                    return None, None, None
                elif no_rain_cnt < rain_counts_dict[0]:
                    target_1d = targets[0].cpu().numpy().flatten()

                    for idx in np.random.choice(np.where(target_1d == 0)[0], rain_counts_dict[0] - no_rain_cnt):
                        target_1d[idx] = -9999

                    targets = torch.from_numpy(target_1d).view(targets_shape).cuda()
            else:
                # Counts rain points
                unique, counts = np.unique(targets[0].cpu().numpy(), return_counts=True)
                rain_counts_dict = dict(zip(unique, counts))

                target_1d = targets[0].cpu().numpy().flatten()

                for idx in np.random.choice(np.where(target_1d == 0)[0],
                                            int(rain_counts_dict[0] * (1 - self.args.no_rain_resample_ratio)),
                                            replace=False):
                    target_1d[idx] = -9999

                targets = torch.from_numpy(target_1d).view(targets_shape).cuda()

        stn_codi = self.remove_missing_station(targets)

        stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]

        pred_labels, target_labels = self.get_stat(stn_preds, stn_targets, mode=mode)

        loss = F.cross_entropy(stn_preds, stn_targets, reduction='none')
        loss = torch.mean(torch.mean(loss, dim=1))

        return loss, pred_labels, target_labels
    
class DiceLoss(nn.Module):
    def __init__(self, args, device, num_classes, balance, experiment_name=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.reference = args.reference
        self.alpha = 0.75
        self.device = device
        self.balance = balance

        
    def forward(self, pred_labels, target_labels, device):
        confusion_matrix = Variable(torch.zeros((self.num_classes, self.num_classes)), requires_grad=True).to(self.device)
    
        for i in range(pred_labels.shape[1]):
            confusion_matrix[target_labels[0, i], pred_labels[0, i]] += (target_labels[0, i]==pred_labels[0, i]).sum()
        
        dice = 0.0
        for clas_num in range(1, self.num_classes):
            tp, fn, fp = confusion_matrix[clas_num, clas_num], confusion_matrix[clas_num, :].sum()-confusion_matrix[clas_num, clas_num], confusion_matrix[:, clas_num].sum()-confusion_matrix[clas_num, clas_num]
            dice += (2 * tp) / (2*tp + fn + fp + 1e-6)
        dice /= (self.num_classes-1)

        return self.balance * (1 - dice ** self.alpha)
