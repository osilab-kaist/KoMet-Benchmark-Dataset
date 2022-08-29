## 📣 KoMet.v1.1

> **We have updated the dataset to include data from 9/21/2020 to 6/20/2021. Download instructions are up now, and statistics will be updated shortly.**

# 🌧 KoMet-Benchmark-Dataset


> Benchmark Dataset for Precipitation Forecasting by Post-Processing the Numerical Weather Prediction

This repository contains the data and code to reproduce all the analyses in the paper ([Link](https://arxiv.org/abs/2206.15241)). If you need something immediately
or find it confusing, please open a GitHub issue or email us. We recommend reading the paper, appendix, and below
descriptions thoroughly before running the code. Future code modifications and official developments will take place
here.

Paper: *"Benchmark Dataset for Precipitation Forecasting by
Post-Processing the Numerical Weather Prediction."*, under review in the NeurIPS 22 Benchmark Dataset Track

# 📔️ Overview

We briefly describe our KoMet Dataset in this section, but **we highly recommend reading Section 3 of the paper.**

## 📝 Dataset Specification

The **KoMet** Dataset (provided by [National Institute of Meteorogical Sciences](http://www.nimr.go.kr/MA/main.jsp), of
the
[Korea Meteorogical Administration (KMA)](https://www.kma.go.kr/home/index.jsp). This dataset is comprised of
**GDAPS-KIM**, a global numerical weather prediction model operated by the KMA, as well as **Automatic Weather Station
(AWS)** observations which serve as ground-truth precipitation data.

Using the dataset, **the main goal is to post-process the GDAPS-KIM output to yield a refined precipitation forecast by
using a deep neural networks**. Here, the deep model is trained with supervision, using AWS observations as ground-truth
labels.

The KoMet dataset has records from **July 1st to August 31st of 2020 and 2021**. Due to the seasonal characteristics of
Korea, the frequency of rainfall is intensive in summer (i.e., from July to August), while it rarely rains in other
seasons. Specifically, GDAPS-KIM included in our dataset contains daily predictions executed at 00:00 UTC leading up to
87 hours in the future, containing 17 atmospheric variables, consisting of 5 Pres variables at 22 different isobraic
surfaces, and 12 Unis variables. All values are real-numbered and provided in double precision floating point format,
following the source data. AWS is provided with all available hourly precipitation data for the specified period. More
precisely, for each year, observations are included until September 3rd, 14:00 UTC, which corresponds to the final
GDAPS-KIM outputs made on August 31st 00:00 UTC with lead time of 87 hours. We provide detailed information on the
atmospheric variables as well as data sources in the paper.

## 🔢 Data Interface (for Models)

- **Input**: **GDAPS-KIM** is an input presented in array format. Before the propagation, the normalization modules acts
  in a feature-wise manner, linearly scaling the features based on min-max values derived from the entire dataset.
- **Output**: We formulate ther precipitation calibration task as a pointwise classification task pertaining to three
  classes: 'non-rain', 'rain', and 'heavy rain'. Below table shows the statistics regarding the frequency of each class.
  Following this, the AWS observation data is pre-processed into 2D array format according to the grids used in
  GDAPS-KIM, respectively. The location of each station is determined within each grid based on the location metadata of
  AWS stations and grid specifications for KIM.

  | Rain rate (mm/h) | Proportion (%) | Rainfall Level |
  | :-------------: | :-------------: | :-------------: |
  | [0.0, 0.1)               |  87.24  | Non-Rain |
  | [0.1, 1.0)               |  11.59 | Rain |
  | [1.0, infty)               |  1.19  | Heavy Rain |

## 📐 Dataset Split

We split the data temporally into three non-overlapping datasets by repeatedly using approximately 4 days for training
followed by 2 days for validation and 2 days for testing. With reference
to [Sonderby et al.](https://arxiv.org/abs/2003.12140), this category of temporal split is utilized.

This is implemented in the `cyclic_split()` function in `data/data_split.py`, which returns three `Subset` instances,
following standard PyTorch split functions.

# 🚀 Getting Started

## 📁 Dataset Download

1. Download `.tar.gz` files from the following Dropbox folder: https://www.dropbox.com/sh/vbme8g8wtx9pitg/AAAB4o6_GhRq0wMc1JxdXFrVa?dl=0
2. Create folder `nims/` and `nims/GDPS_KIM`
3. Unzip tar files as follows:
  - Unzip `AWS.tar.gz` into `nims/`
  - Unzip `GDAPS_KIM_*.tar.gz` archives into `/nims/GDPS_KIM`

The resulting `nims/` dataset folder should contain the follow:

```
├── AWS/
│  ├── 2020/
│  └── 2021/
├── AWS_GDPS_KIM_GRID/
│  ├── 2020/
│  └── 2021/
├── GDPS_KIM/
│  ├── 202007/
│  ├── 202008/
│  ├── ...
```

Finally, move the `nims/` directory to `/data/nims/` to use the training scripts as-is. If you are unable to
create or access the `/data` directory, you may specify a custom location using the `--dataset_dir` argument. Refer to
`parse_args()` in `utils.py`.

## 🐍 Requirements

The code is currently being developed and tested on Python 3.8 and PyTorch 1.8, as of June 2022.

- Install `torch` and `torchvision` according to the instructions on the
  [PyTorch website](https://pytorch.org/get-started/locally/).
- Install remaining requirements provided in `requirements.txt`, using `pip -r requirements.txt`.

### ⚠️ Setup Local Package (IMPORTANT)

Register the project directory as a Python package to allow for absolute imports.

```bash
python3 setup.py develop
```

# 🛠 Data Classes

We provide two layers of abstraction to facilitate data manipulation.

- `data.base_dataset.BaseDataset`: `BaseDataset` classes provide low-level access to NWP and AWS. Using the
  `load_array()` method, you can fetch individual numpy arrays of NWP predictions or AWS observations corresponding to
  specific datetimes (and lead times, for NWP), without the need to worry about individual data paths or the particular
  format of the underlying data files.
- `data.dataset.StandardDataset`: `StandardDataset` classes are build on top of `BaseDataset` classes, acting as
  iterables over x, y samples for model training. They inherit the standard interface of `torch.utils.data.Dataset`
  classes.

Refer to `notebooks/dataset_example.ipynb` on usage.

## 📝 `StandardDataset` Arguments

Here is a snippet of the `load_dataset_from_args()` convenience method provided in `utils.py`, which is used to
instantiate a `StandardDataset` for training. We briefly describe the arguments below.

```python
from data.dataset import get_dataset_class


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
```

- `input_data`: the type of NWP model. Now, only `gdaps_kim` is supported.
- `utc`: the hour in which NWP prediction was ran in UTC time (data is only provided for 00 UTC)
- `window_size`: how many sequences in one instance. (e.g., 10 is to use 10 hour consecutive sequences in a simulation)
- `root_dir`: base directory for datasets
- `data_intervals`: start and end dates (ex, 2020-07 2021-08)
- `start_lead_time`: start of lead_time (how many hours between origin time and prediction target time) range, inclusive
- `end_lead_time`: end of lead_time (how many hours between origin time and prediction target time) range, exclusive
- `variable_filter`: which variables to use. It is a list of variable name (str type).

# 🔬 Model Development

## 👟 Training

The following is an example snippet from `scripts/unet.sh` for training a vanilla U-Net model.

```bash
python train.py --model="unet" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=20 --normalization \
                --rain_thresholds 0.1 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --custom_name="unet_test"
```

Refer to scripts in `scripts/` for additional examples. Note that `scripts/*_experiments/` contain scripts that launch
multiple training runs, in parallel, via tmux sessions on multiple GPUs.
Run `source scripts/*_experiments/launch_all.sh` to launch them as-is, or refer to the `run.sh` files for usage of CLI
arguments.

For more information on CLI arguments, refer to `parse_args()` in `utils.py`.

## 📋 Evaluation

During training, epoch-wise evaluation results on all data splits are logged in the `output/` directory.

Refer to `notebooks/evaluation_example.ipynb` on how to load and analyze the evaluations, using the provided functions.
You can execute the notebook code yourself after running the example training script `scripts/unet.sh`.

## 🤖 Models

Currently, we support three models from the following papers:

- [U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al. 2015](http://arxiv.org/abs/1505.04597)
- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, Shi et al. 2015](https://arxiv.org/abs/1506.04214)
- [MetNet: A Neural Weather Model for Precipitation Forecasting, Sonderby et al. 2020](https://arxiv.org/abs/2003.12140)

You can load the model using the `set_model()` function in `utils.py`. Below is an example of initializing the
MetNet model with various hyperparameters.

```python
from model.metnet import MetNet

model = MetNet(input_data=input_data,
               window_size=window_size,
               num_cls=num_classes,
               in_channels=in_channels,
               start_dim=start_dim,
               center_crop=False,
               center=None,
               pred_hour=1)
```

# 🏛 Acknowledgements

> This work was funded by the Korea Meteorological Administration
> Research and Development Program "'Development of AI techniques for
> Weather Forecasting" under Grant (KMA2021-00121).
