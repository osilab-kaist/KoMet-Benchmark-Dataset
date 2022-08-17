import datetime
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.evaluate import evaluate_model
from evaluation.metrics import compile_metrics
from paths import *
from utils import save_evaluation_results_for_args

__all__ = ['NIMSTrainer']


class NIMSTrainer:
    def __init__(self, model, criterion, dice_criterion, optimizer, scheduler, device, train_loader, valid_loader, test_loader,
                 experiment_name, args, normalization=None):
        self.args = args

        self.model = model
        self.model_name = args.model
        self.criterion = criterion
        self.dice_criterion = dice_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = args.num_epochs

        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.input_data = args.input_data
        self.reference = args.reference
        self.experiment_name = experiment_name
        self.normalization = normalization
        self.rain_thresholds = args.rain_thresholds

        self.trained_weight = {'model': None,
                               'model_name': args.model,
                               'n_blocks': args.n_blocks,  # U-Net related
                               'start_channels': args.start_channels,  # U-Net related
                               'no_residual': args.no_residual,  # U-Net related
                               'no_skip': args.no_skip,  # U-Net related
                               'use_tte': args.use_tte,  # U-Net related
                               'num_layers': args.num_layers,  # ConvLSTM related
                               'hidden_dim': args.hidden_dim,  # ConvLSTM related
                               'kernel_size': args.kernel_size,  # ConvLSTM related
                               'start_dim': args.start_dim,  # MetNet related
                               'window_size': args.window_size,
                               'model_utc': args.model_utc,
                               'dry_sampling_rate': args.dry_sampling_rate,
                               'global_sampling_rate': args.global_sampling_rate,
                               'num_epochs': args.num_epochs,
                               'batch_size': args.batch_size,
                               'optimizer': args.optimizer,
                               'lr': args.lr,
                               'wd': args.wd,
                               'custom_name': args.custom_name,
                               'rain_thresholds': args.rain_thresholds,
                               'num_classes': args.num_classes,
                               'norm_max': normalization['max_values'] if normalization else None,
                               'norm_min': normalization['min_values'] if normalization else None}

        self.device_idx = int(args.device)
        self.model.to(self.device)

        # Hotfix - intermediate evaluation on test set
        self.intermediate_test = args.intermediate_test
        if args.intermediate_test:
            print("Intermediate evaluation on test set is enabled")
        else:
            print("Omitting intermediate evaluation on test set")

        self.log_dict = {
            'complete': False,
            'history': [],
        }
        self.log_json_path = get_train_log_json_path(self.experiment_name, makedirs=True)
        self.log_txt_path = get_train_log_txt_path(self.experiment_name, makedirs=True)
        self.history_path = get_train_log_history_path(self.experiment_name, makedirs=True)

    def save_trained_weight(self, epoch: int = None, step: int = None):
        if sum([epoch is not None, step is not None]) != 1:
            raise ValueError('Only one of `epoch` or `step` must be specified to save model')
        trained_weight_path = get_trained_model_path(self.experiment_name, epoch=epoch, step=step, makedirs=True)
        if os.path.isfile(trained_weight_path):
            os.remove(trained_weight_path)
        torch.save(self.trained_weight, trained_weight_path)

    def train(self):
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            epoch_log_dict = dict()
            epoch_log_dict['epoch'] = epoch

            epoch_log = 'Epoch {:3d} / {:3d} (GPU {})'.format(epoch, self.num_epochs, self.device_idx)
            self._log(epoch_log.center(40).center(80, '='))

            # Run training epoch
            loss, confusion, metrics = self._epoch(self.train_loader, mode='train')
            correct = confusion[np.diag_indices_from(confusion)].sum()
            accuracy = (correct / confusion.sum()).item()
            epoch_log_dict['loss'] = loss
            epoch_log_dict['accuracy'] = accuracy

            # Save model
            self.trained_weight['model'] = self.model.state_dict()
            self.save_trained_weight(epoch=epoch, step=None)

            # Save/log train evaluation metrics
            summary = save_evaluation_results_for_args(confusion, metrics, epoch, self.args, "train", loss=loss)
            self._log('Train Metrics '.center(40).center(80, '#'))
            self._log(summary)

            self.log_dict['history'].append(epoch_log_dict)

            # Validation evaluation
            _, loss, confusion, metrics = evaluate_model(self.model, self.valid_loader, self.args.rain_thresholds,
                                                         self.criterion, self.device, self.normalization)
            summary = save_evaluation_results_for_args(confusion, metrics, epoch, self.args, "val", loss=loss)
            self._log('Validation Metrics '.center(40).center(80, '#'))
            self._log(summary)

            # Test evaluation
            _, loss, confusion, metrics = evaluate_model(self.model, self.test_loader, self.args.rain_thresholds,
                                                         self.criterion, self.device, self.normalization)
            summary = save_evaluation_results_for_args(confusion, metrics, epoch, self.args, "test", loss=loss)
            self._log('Test Metrics '.center(40).center(80, '#'))
            self._log(summary)

            with open(self.log_json_path, 'w') as f:
                json.dump(self.log_dict, f)

            history = pd.DataFrame(self.log_dict['history'])
            history.to_csv(self.history_path)

            if self.scheduler:
                self.scheduler.step()

        # Save history
        self.log_dict['complete'] = True
        with open(self.log_json_path, 'w') as f:
            json.dump(self.log_dict, f)

        print('Log files saved to the following paths:')
        print(self.log_txt_path)
        print(self.log_json_path)
        print(self.history_path)

    def _log(self, text):
        print(text)
        with open(self.log_txt_path, 'a') as f:
            f.write(text + '\n')

    def _epoch(self, data_loader, mode):
        """
        :param data_loader:
        :param mode:
        :return:
        """
        pbar = tqdm(data_loader)
        total_loss = 0
        total_samples = 0

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int)
        metrics_by_threshold = defaultdict(list)  # metrics_by_threshold[threshold][step]: DataFrame
        for i, (images, target, t) in enumerate(pbar):
            # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
            timestamps = []
            for e in t:
                origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
                lead_time = e[4].item()
                timestamps.append((origin, lead_time))

            if self.normalization:
                with torch.no_grad():
                    for i, (max_val, min_val) in enumerate(zip(self.normalization['max_values'],
                                                               self.normalization['min_values'])):
                        if min_val < 0:
                            images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                        else:
                            images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

            images = images.type(torch.FloatTensor).to(self.device)
            target = target.type(torch.LongTensor).to(self.device)
            output = self.model(images, t)

            # Note, pred_labels will be removed soon
            loss, pred_labels, target_labels = self.criterion(output, target, timestamps, mode=mode)
            if self.dice_criterion !=None:
                loss += self.dice_criterion(pred_labels, target_labels, self.device)
            _, predictions = output.detach().cpu().topk(1, dim=1, largest=True,
                                                        sorted=True)  # (batch_size, height, width)
            step_confusion, step_metrics_by_threshold = compile_metrics(data_loader.dataset, predictions.numpy(),
                                                                        timestamps, self.args.rain_thresholds)
            confusion_matrix += step_confusion
            for threshold, metrics in step_metrics_by_threshold.items():
                metrics_by_threshold[threshold].append(metrics)

            if loss is None:
                continue
            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
        average_loss = total_loss / total_samples

        return average_loss, confusion_matrix, metrics_by_threshold
