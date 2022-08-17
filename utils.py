import argparse
import os
import random
import time
from collections import namedtuple
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, cpu_count
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from data.dataset import get_dataset_class
from evaluation.metrics import compute_evaluation_metrics
from losses import *
from model.conv_lstm import ConvLSTM
from model.metnet import MetNet
from model.unet_model import UNet
from paths import PROJECT_ROOT, OUTPUT_DIR, get_summary_path, get_binary_metrics_path, get_confusion_matrix_path

try:
    import setproctitle
except ImportError:
    setproctitle = None

__all__ = ['NIMSStat', 'parse_args', 'set_device', 'fix_seed',
           'set_model', 'set_optimizer', 'get_experiment_name', 'get_min_max_values',
           'list_experiments', 'get_experiments', 'select_experiment', 'parse_date', 'PROJECT_ROOT',
           'save_evaluation_results_for_args', 'load_dataset_from_args']

NIMSStat = namedtuple('NIMSStat', 'acc, csi, pod, far, f1, bias')
MONTHLY_MODE = 1
DAILY_MODE = 2

LIST_NUM_MODE = 1
PRETRAIN_NAME_MODE = 2


def parse_args(manual_input=None):
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    common = parser.add_argument_group('common')
    common.add_argument('--model', default='unet', type=str, help='which model to use [unet, attn_unet, convlstm]')
    common.add_argument('--dataset_dir', default='/data/nims', type=str, help='root directory of dataset')
    common.add_argument('--seed', default=0, type=int, help='seed number')
    common.add_argument('--input_data', default=None, type=str, help='input data: gdaps_kim, gdaps_um, ldaps')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--num_workers', default=5, type=int, help='# of workers for dataloader')
    common.add_argument('--date_intervals', default=['2020-07', '2020-08', '2021-07', '2021-08'], nargs='+',
                        help='list of date intervals: start and end dates in YYYY-MM form, inclusive')
    common.add_argument('--start_lead_time', type=int, default=6,
                        help='start of lead_time (how many hours between origin time and prediction target time) range, inclusive')
    common.add_argument('--end_lead_time', type=int, default=88,
                        help='end of lead_time (how many hours between origin time and prediction target time) range, exclusive')
    common.add_argument('--intermediate_test', action='store_true', help='evaluate on test set during training')
    common.add_argument('--intermediate_test_step_interval', default=None, type=int,
                        help='evaluate on test set during training within epochs at the given step interval')
    common.add_argument('--discard_predictions', type=bool, default=True,
                        help='whether to delete all predictions (excluding the last epoch) after intermediate evaluation')

    common.add_argument('--prediction_start_date', default='2020-08-01', type=str,
                        help='prediction start date in YYYY-MM or YYYY-MM-DD form, inclusive')
    common.add_argument('--prediction_end_date', default='2020-08-14', type=str,
                        help='prediction end date in YYYY-MM or YYYY-MM-DD form (end of month for format 1), inclusive')
    common.add_argument('--prediction_epoch', default=None, type=int,
                        help='generate predictions from model trained until the specified epoch')
    common.add_argument('--prediction_step', default=None, type=int,
                        help='generate predictions from model trained until the specified step')
    common.add_argument('--rain_thresholds', default=[0.1, 10.0], type=float, nargs='+',
                        help='thresholds for rainfall classes')
    # common.add_argument('--noise_prob', default=0.0, type=float, help='probability of adding noise to original data')
    common.add_argument('--variable_filter', type=str, help='variable selection filter for dataset')
    common.add_argument('--custom_name', default=None, type=str, help='add customize experiment name')
    common.add_argument('--experiment_name', default=None, type=str,
                        help='experiment name used for gen_nc')  # should only be used for gen_nc
    # common.add_argument('--debug', help='turn on debugging print', action='store_true')
    common.add_argument('--interpolate_aws', default=False, action="store_true")

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--embedding_dim', default=8, type=int, help='dimension of embedding of per time')
    unet.add_argument('--n_blocks', default=3, type=int, help='# of blocks in Down and Up phase')
    unet.add_argument('--start_channels', default=16, type=int, help='# of channels after first block of unet')
    unet.add_argument('--no_residual', default=False, action='store_true',
                      help='do not use inner block residual connection')
    unet.add_argument('--no_skip', default=[], type=int, nargs='+',
                      help='indices of Unet blocks to omit skip connection')
    unet.add_argument('--use_tte', default=False, action='store_true', help='use target time embedding')

    convlstm = parser.add_argument_group('convlstm related')
    convlstm.add_argument('--hidden_dim', default=16, type=int, help='hidden dimension in ConvLSTM')
    convlstm.add_argument('--num_layers', default=3, type=int, help='# of layers in ConvLSTM')
    convlstm.add_argument('--kernel_size', default=(3, 3), type=int, nargs=2, help='kernel size in ConvLSTM')

    metnet = parser.add_argument_group('metnet related')
    metnet.add_argument('--start_dim', default=16, type=int, help='start dimension in MetNet')

    nims_dataset = parser.add_argument_group('nims dataset related')
    nims_dataset.add_argument('--window_size', default=3, type=int, help='# of input sequences in time')
    nims_dataset.add_argument('--model_utc', default=0, type=int, help='base UTC time of data (0, 6, 12, 18)')
    nims_dataset.add_argument('--normalization', default=False, help='normalize input data', action='store_true')
    nims_dataset.add_argument('--reference', default=None, type=str, help='which data to be used as a ground truth')

    hyperparam = parser.add_argument_group('hyper-parameters')
    hyperparam.add_argument('--num_epochs', default=40, type=int, help='# of training epochs')
    hyperparam.add_argument('--batch_size', default=1, type=int, help='batch size')
    hyperparam.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')
    hyperparam.add_argument('--disable_nesterov', dest="nesterov", action="store_false",
                            help='disable nesterov momentum for sgd')
    hyperparam.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer')
    hyperparam.add_argument('--momentum', default=0.0, type=float, help='momentum')
    hyperparam.add_argument('--wd', default=0, type=float, help='weight decay')

    sampling = parser.add_argument_group('sampling')
    sampling.add_argument('--dry_sampling_rate', default=1.0, type=float,
                          help='(Under)sample dry point by given fixed rate')
    sampling.add_argument('--global_sampling_rate', default=1.0, type=float,
                          help='(Under)sample all points by given fixed rate')
    sampling.add_argument('--no_rain_ratio', default=None, type=float,
                          help="(Under)sample dry points until `dry:rain` meets the given ratio. Ignore if there are insufficient dry points.")
    sampling.add_argument('--rain_ratio', default=None, type=float,
                          help="ratio of precipitation for heavy precipitation undersampling")
    sampling.add_argument('--target_precipitation', default="rain", type=str,
                          help="rain_ratio target class for binary classification")

    # augmentation = parser.add_argument_group('augmentation')
    # augmentation.add_argument('--aug_method', default=None, type=str, help='which method to use (mixup, add_noise)')
    # augmentation.add_argument('--num_aug_data', default=100, type=int, help='number of augmented data points')
    # augmentation.add_argument('--aug_rain_threshold', default=10.0, type=float,
    #                           help='threshold for filtering augmentation dataset')
    # augmentation.add_argument('--aug_rain_cnt_threshold', default=10, type=int,
    #                           help='number of points threshold for filtering augmentation dataset')

    # augmentation.add_argument('--aug_noise_scale', default=0.01, type=float,
    #                           help='noise scale for add_noise data augmentation')
    # augmentation.add_argument('--add_nose_transform', action='store_true', help='add gaussian noise to every input')

    nc_gen = parser.add_argument_group('nc_gen')
    common.add_argument('--realtime', default=False, action="store_true", help="For realtime output")
    nc_gen.add_argument('--date', default=None, type=str, help='Date for netCDF file')

    args = parser.parse_args(manual_input)

    # Post-parse
    args.num_classes = len(args.rain_thresholds) + 1

    assert len(args.date_intervals) % 2 == 0
    start_dates = [parse_date(d, end=False) for d in args.date_intervals[::2]]
    end_dates = [parse_date(d, end=True) for d in args.date_intervals[1::2]]
    args.date_intervals = list(zip(start_dates, end_dates))

    for i, (start, end) in enumerate(args.date_intervals):
        if end - start < timedelta(hours=1):
            starts = start.strftime("%Y-%m-%d")
            ends = end.strftime("%Y-%m-%d")
            raise ValueError("{}th date interval is invalid: {} - {}".format(i, starts, ends))

    assert args.input_data in ['ldaps', 'gdaps_um', 'gdaps_kim'], \
        'input_data must be one of [ldaps, gdaps_um, gdaps_kim]'

    assert args.model_utc in [0, 6, 12, 18], \
        'model_utc must be one of [0, 6, 12, 18]'

    assert args.reference in ['aws', 'reanalysis', None], \
        'reference must be one of [aws, reanalysis]'

    return args


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset_from_args(args, **kwargs):
    """
    **kwargs include transform, target_transform, etc.
    """
    dataset_class = get_dataset_class(args.input_data)
    return dataset_class(utc=args.model_utc,
                         window_size=args.window_size,
                         root_dir=args.dataset_dir,
                         date_intervals=args.date_intervals,
                         start_lead_time=args.start_lead_time,
                         end_lead_time=args.end_lead_time,
                         variable_filter=args.variable_filter,
                         **kwargs)


def generate_evaluation_summary(confusion: np.ndarray, metrics_by_threshold: Dict[float, pd.DataFrame], loss=None):
    accuracy = confusion[np.diag_indices_from(confusion)].sum() / confusion.sum()
    fmt = '{:20s} {:>7.4f}     '

    lines = [''] * 4
    lines[0] = fmt.format('acc', accuracy)
    if loss is not None:
        lines[0] += fmt.format('loss', loss)

    for t, name in zip([0.1, 10.0], ['rain', 'heavy_rain']):
        if t not in metrics_by_threshold:
            continue
        metrics = metrics_by_threshold[t].sum()
        compute_evaluation_metrics(metrics)
        lines[1] += fmt.format(name + '_acc', metrics.acc)
        lines[2] += fmt.format(name + '_csi', metrics.csi)
        lines[3] += fmt.format(name + '_bias', metrics.bias)

    return '\n'.join(lines)


def save_evaluation_results_for_args(confusion: np.ndarray, metrics_by_threshold: Dict[float, pd.DataFrame], epoch,
                                     args, subset='train', loss=None, verbose=True) -> str:
    saved_paths = []

    # Save confusion matrix
    path = get_confusion_matrix_path(args.experiment_name, epoch, subset)
    saved_paths.append(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, confusion)

    # Save binary metrics (for rain, heavy_rain if they exist)
    for t, name in zip([0.1, 10.0], ["rain", "heavy"]):
        if t not in metrics_by_threshold:
            continue
        metrics = metrics_by_threshold[t]
        path = get_binary_metrics_path(args.experiment_name, epoch, subset, t)
        saved_paths.append(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metrics.to_csv(path)

    # Save summary
    summary = generate_evaluation_summary(confusion, metrics_by_threshold, loss=loss)
    path = get_summary_path(args.experiment_name, epoch, subset)
    saved_paths.append(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(summary)

    if verbose:
        print("Saved evaluation results to:")
        for p in saved_paths:
            print(p)

    return summary


def set_device(args):
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda')

    return device


def set_model(sample, device, args,
              experiment_name=None, finetune=False, model_path=None):
    # Create a model and criterion
    if args.model == 'unet':
        model = UNet(input_data=args.input_data,
                     window_size=args.window_size,
                     embedding_dim=args.embedding_dim,
                     n_channels=sample.shape[0] * sample.shape[1],
                     n_classes=args.num_classes,
                     n_blocks=args.n_blocks,
                     start_channels=args.start_channels,
                     batch_size=args.batch_size,
                     end_lead_time=args.end_lead_time,
                     residual=not args.no_residual,
                     no_skip=args.no_skip,
                     use_tte=args.use_tte)
    elif args.model == 'convlstm':
        model = ConvLSTM(input_data=args.input_data,
                         window_size=args.window_size,
                         input_dim=sample.shape[1],
                         hidden_dim=args.hidden_dim,
                         kernel_size=tuple(args.kernel_size),  # hotfix: only supports single tuple of size 2
                         num_layers=args.num_layers,
                         num_classes=args.num_classes,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False)
    elif args.model == 'metnet':
        model = MetNet(input_data=args.input_data,
                       window_size=args.window_size,
                       num_cls=args.num_classes,
                       in_channels=sample.shape[1],
                       start_dim=args.start_dim,
                       center_crop=False,
                       center=None,
                       pred_hour=1)
    elif args.model == 'point':
        model = precipitation_point(input_data=args.input_data,
                       window_size=args.window_size,
                       num_cls=args.num_classes,
                       in_channels=sample.shape[1],
                       start_dim=args.start_dim,
                       center_crop=False,
                       center=None,
                       pred_hour=1)
    else:
        raise ValueError('{} is not a valid argument for `args.model`'.format(args.model))

    criterion = CrossEntropyLoss(args=args,
                                 device=device,
                                 num_classes=args.num_classes,
                                 experiment_name=experiment_name)

    if finetune:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)

    # model = DataParallel(model)
    return model, criterion


def set_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.wd, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  alpha=0.9, eps=1e-6)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 30)

    return optimizer, scheduler


def get_experiment_name(args):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    fmt = '{input_data}_{model}_utc{utc:02d}_{custom_name}'
    custom_name = args.custom_name
    if not custom_name:
        custom_name = datetime.now().strftime('untitled_%y%m%d_%H%M%S')

    experiment_name = fmt.format(input_data=args.input_data,
                                 model=args.model,
                                 utc=args.model_utc,
                                 custom_name=custom_name
                                 )

    if setproctitle:
        setproctitle.setproctitle(experiment_name)

    return experiment_name


def _get_min_max_values(dataset, indices, queue=None):
    '''
        Return min and max values of ldaps_inputs in train
            Args : train_dataset
                (nwp_input, gt, target_time_tensor = train_dataset[i])
            Returns :
                max_values : max_values [features, ]
                min_values : min_values [features, ]
    '''

    max_values = None
    min_values = None

    # Check out training set
    for i, idx in enumerate(indices):
        # Pop out data
        nwp_input, _, _ = dataset[idx]
        if type(nwp_input) == torch.Tensor:
            nwp_input = nwp_input.numpy()
        nwp_input = np.transpose(nwp_input, axes=(1, 0, 2, 3))
        nwp_input = np.reshape(nwp_input, (nwp_input.shape[0], -1))

        # Evaluate min / max on current data
        temp_max = np.amax(nwp_input, axis=-1)
        temp_min = np.amin(nwp_input, axis=-1)

        # Edge case
        if i == 0:
            max_values = temp_max
            min_values = temp_min

        # Comparing max / min values
        max_values = np.maximum(max_values, temp_max)
        min_values = np.minimum(max_values, temp_min)

    if queue:
        queue.put((max_values, min_values))
    else:
        return max_values, min_values


def get_min_max_values(dataset):
    # Make indices list
    indices = list(range(len(dataset)))

    num_processes = cpu_count() // 4
    num_indices_per_process = len(indices) // num_processes

    # Create queue
    queues = []
    for i in range(num_processes):
        queues.append(Queue())

    # Create processes
    processes = []
    for i in range(num_processes):
        start_idx = i * num_indices_per_process
        end_idx = start_idx + num_indices_per_process

        if i == num_processes - 1:
            processes.append(Process(target=_get_min_max_values,
                                     args=(dataset, indices[start_idx:],
                                           queues[i])))
        else:
            processes.append(Process(target=_get_min_max_values,
                                     args=(dataset, indices[start_idx:end_idx],
                                           queues[i])))

    # Start processes
    for i in range(num_processes):
        processes[i].start()

    # Join processes
    animation = "|/-\\"
    idx = 0
    alive_flag = [True] * num_processes
    while True:
        for i in range(num_processes):
            processes[i].join(timeout=0)
            if not processes[i].is_alive():
                alive_flag[i] = False

        if True not in alive_flag:
            print()
            break

        print('Normalization Start. Please Wait...{}'.format(animation[idx % len(animation)]), end='\r')
        idx += 1
        time.sleep(0.1)

    print('Normalization End!')
    print()

    # Get return value of each process
    max_values, min_values = None, None
    for i in range(num_processes):
        proc_result = queues[i].get()

        if i == 0:
            max_values = proc_result[0]
            min_values = proc_result[1]
        else:
            max_values = np.maximum(max_values, proc_result[0])
            min_values = np.minimum(min_values, proc_result[1])

    # Convert to PyTorch tensor
    max_values = torch.tensor(max_values)
    min_values = torch.tensor(min_values)

    return max_values, min_values


def get_experiments() -> List[Tuple[str, datetime]]:
    """
    Get experiment names and their last modified times.
    This simply searches the subdirectory names in `results/`

    :return: [
        (name, modified_time)
    ]
    """
    experiment_names = sorted([f for f in os.listdir(OUTPUT_DIR)])

    experiments = []
    for name in experiment_names:
        modified = datetime.fromtimestamp(os.path.getmtime(os.path.join(OUTPUT_DIR, name)))
        experiments.append((name, modified))

    experiments.sort(key=lambda e: e[1])
    return experiments


def list_experiments(experiments: List[Tuple[str, datetime]] = None):
    if experiments is None:
        experiments = get_experiments()

    print('-' * 100)
    print('{:4s}  {:19s}  {}'.format('Idx', 'Last Modified', 'Experiment'))
    print('-' * 100)
    for i, (name, modified) in enumerate(experiments):
        path_date = modified.strftime('%Y/%m/%d %H:%M:%S')
        print('{:>4d}  {:s}  {}'.format(i + 1, path_date, name))


def select_experiment() -> str:
    experiments = get_experiments()

    print('Which experiment do you like to test?'.center(50).center(100, '='))
    list_experiments(experiments)

    while True:
        try:
            choice = int(input('\n> '))
            name, modified = experiments[choice - 1]
            return name
        except (IndexError, ValueError):
            print('Invalid choice. Try again.')


def parse_date(date_string: str, end: bool) -> datetime:
    """
    날짜 string을 parse합니다. 다음과 같은 형태를 지원합니다.

    - 20-08 -> 2020/08/01 00:00
    - 20-08-15 -> 2020/08/15 00:00

    end=True일 경우, YY-MM 형태로 입력이 들어오면 그 달의 마지막 일자 선택됩니다.

    - 2020-08 -> 2020/08/31 00:00
    - 2020-08-15 -> 2020/08/15 00:00

    :param date_string:
    :param end:
    :return:
    """
    try:  # %Y-%m-%d
        dt = datetime.strptime(date_string, '%Y-%m-%d')
        return dt
    except:
        pass

    try:  # %Y-%m
        dt = datetime.strptime(date_string, '%Y-%m')
        if end:
            if dt.month == 12:
                dt = dt.replace(year=dt.year + 1, month=1)
            else:
                dt = dt.replace(month=dt.month + 1)
            dt -= timedelta(days=1)
        return dt
    except:
        pass

    raise ValueError("{} is not a valid date string".format(date_string))
