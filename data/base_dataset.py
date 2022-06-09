import glob
import os
import re
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import netCDF4 as nc
import numpy as np

__all__ = ['BaseDataset', 'GdapsKimBaseDataset', 'GdapsUmBaseDataset', 'AwsBaseDataset', 'AwsBaseDatasetForGdapsKim',
           'AwsBaseDatasetForGdapsUm']


class BaseDataset:
    """
    다양한 수치모델 (NWP)과 AWS 데이터를 각각 ndarray 형태로 쉽게 접근하기 위한 추상 클래스입니다. 주요 역할은 다음과 같습니다:

    - 주어진 데이터 관련 클래스 속성을 바탕으로 데이터 경로를 모두 파악하고 `self.data_path_dict`에 dictionary 형태로 정리합니다.
      이때 사용되는 클래스 속성은 다음과 같습니다:

      - `data_path_glob`: 데이터 path를 모두 찾아내는 glob expression (glob 모듈 참고)
      - `data_path_regex`: 데이터 path에서 year, month, day, hour, lead_time 정보를 추출하는 정규표현식

    - `load_array()` 함수의 인자로 NWP 예측 (혹은 AWS 관측) 시점만 넘겨주면 해당 예측/관측 데이터를 ndarray 형태로 불러옵니다.

    ## 용어

    - `origin`: 수치모델에서 예측을 수행하는 현재 시점을 말하며, `datetime(year, month, day, hour=utc)`로 결정됩니다.
    - `lead_time`: 수치모델에서 `origin` 기준 몇시간 후를 예측하고자 하는지를 말합니다.
    - `target`: 수치모델에서 예측하고자 하는 시점을 말하며, `origin + timedelta(hours=lead_time)`로 결정됩니다.

    AWS 데이터의 경우, 현재 관측값만 존재하므로 `lead_time=0`을 사용하여 `origin` 시점에서 현재 관측한 값을 불러올 수 있습니다.

    ## 사용 예시 (GdapsKimBaseDataset)

    ```python
    utc = 6
    ds = GdapsKimBaseDataset(root_dir='/home/osilab12/ssd4', variable_filter='rain, snow')
    feature_map = ds.load_array(origin=datetime(2020, 7, 1, hour=utc), lead_time=9)
    plt.imshow(feature_map[0])  # map of rain at 2020/07/01 3:00PM predicted at 6:00AM
    ```

    ## 주요 속성

    - `data_path_dict`: 각각의 target 시점을 그에 해당하는 데이터 path로 대응시키는 dictionary입니다.

      - key: `(origin: datetime, lead_time: int)` 형태의 tuple로, target 시점을 가르킵니다.
      - value: `List[str]` 형태로, key가 가르키는 시점에 해당하는 데이터를 담은 파일 path를 리스트로 저장합니다.

        - Q. 왜 리스트 형태?
        - A. 예를 들어, GDAPS-KIM 모델에서는 단일면, 등압면 데이터가 각기 다른 파일에 저장되어 있기 때문에, 특정 target 시점에 해당하는 path가
             두 개씩 존재합니다. 또 다른 예로, GDAPS-UM 모델에서는 예측값이 모두 하나의 파일에 저장되기 때문에 path가 하나만 존재합니다.
    """
    dataset_dir = None  # e.g., 'GDPSKIM'
    data_path_glob: str = None  # e.g., '**/*.nc'
    data_path_regex: str = None  # regex with named groups: 'year', 'month', 'day', 'hour', 'lead_time'

    def __init__(self, root_dir: str, variable_filter: str = None):
        """
        :param root_dir: `dataset_dir`을 담고 있는 최상위 경로
        :param variable_filter: 수치모델에서 불러오고자 하는 변수를 지정하는 string입니다. string의 형태는 자식 클래스의
          `load_array` 구현에 따라 달라집니다. E.g., `GdapsKimBaseDataset`에서는 comma-separated string 형태로 변수명을
          지정하면 됩니다.
        """
        self._verify_class_attributes()
        self.root_dir = root_dir
        self.variable_filter = variable_filter

        # Search paths and populate `self.data_path_dict`
        self._re_path = re.compile(self.data_path_regex)
        self.data_path_dict = self._generate_data_path_dict()
        self._verify_data_path_dict(self.data_path_dict)
        self.timestamps = list(self.data_path_dict.keys())

    _ret = Union[np.ndarray, Tuple[np.ndarray, List[str]]]

    @abstractmethod
    def load_array(self, origin: datetime, lead_time: int, return_variables: bool = False) -> _ret:
        """
        수치모델에서 origin 시간 기준 lead_time 시간 후에 예측한 feature map을 ndarray 형태로 반환합니다. AWS의 경우,
        `lead_time=0`만 사용 가능하며, origin 시간 기준 관측된 정보를 반환합니다.

        본 함수는 추상 함수로, 자식 클래스에서 개별 데이터 파일의 형태에 맞추어 구현해야 합니다. 주로 `data_path_dict`를 참조하여 해당 시점의
        데이터 파일 path(들)을 얻고 파일을 파싱하여 ndarray 형태로 뽑아냅니다.

        :param origin: 수치모델에서 예측을 실행하는 시점을 말합니다. I.e., `datetime(year, month, day, utc)`.
          GMT 시간대 기준으로 사용 요망.
        :param lead_time: 수치모델에서 origin 기준 몇시간 후를 예측하고자 하는지 지정합니다. AWS의 경우, `lead_time=0`만 사용 가능.
        :param return_variables: FxHxW 형태의 출력 array에서 각각의 F dimension에 해당하는 변수 이름을 리스트 형태로
            추가로 반환할지 여부. True 시, `(feature_map: np.ndarray, variables: List)` 형태로 출력이 됩니다.
        :return:
        """
        raise NotImplementedError

    def load_window(self, origin: datetime, lead_time: int, window_size: int, zero_padding=False) -> np.ndarray:
        arrays = []
        assert lead_time >= 0
        for i in range(window_size):  # fill from end to start
            if lead_time - i < 0:  # padding
                if zero_padding:
                    arrays.insert(0, np.zeros_like(arrays[-1]))
                else:
                    raise Exception('`window_size` must be <= `lead_time + 1` when `zero_padding=False')
            else:
                arrays.insert(0, self.load_array(origin, lead_time - i))
        return np.stack(arrays, axis=0)

    def _verify_data_path_dict(self, data_path_dict):
        pass

    def _generate_data_path_dict(self) -> Dict[Tuple[datetime, int], List[str]]:
        all_paths = glob.glob(os.path.join(self.root_dir, self.dataset_dir, self.data_path_glob), recursive=True)
        all_paths.sort()
        data_path_dict: Dict[Tuple[datetime, int], List[str]] = defaultdict(list)
        for path in all_paths:
            origin_datetime, lead_time = self._parse_data_path(path)
            data_path_dict[(origin_datetime, lead_time)].append(path)
        return data_path_dict

    def _verify_class_attributes(self):
        try:
            assert self.dataset_dir is not None
            assert self.data_path_regex is not None
            assert self.data_path_glob is not None
        except AssertionError:
            raise NotImplementedError('Required class attributes have not been specified')

    def _parse_data_path(self, path):
        match = self._re_path.match(path)
        year = int(match.group('year'))
        month = int(match.group('month'))
        day = int(match.group('day'))
        hour = int(match.group('hour'))
        lead_time = int(match.group('lead_time'))
        origin_time = datetime(year=year, month=month, day=day, hour=hour)
        return origin_time, lead_time

    def _print_data_paths(self, n=10):
        items = list(self.data_path_dict.items())
        items = items[n] if n else items
        for dt, paths_by_lead_time in items:
            for lead_time, paths in paths_by_lead_time.items():
                print('{} (lead_time={})'.format(dt, lead_time))
                for p in paths:
                    print('\t' + p)


class GdapsKimBaseDataset(BaseDataset):
    dataset_dir = 'GDPS_KIM'
    data_path_glob = os.path.join('**', '*.nc')
    data_path_regex = '.*(?P<year>\d{4})(?P<month>\d{2})/(?P<day>\d{2})/(?P<hour>\d{2})/.*ft(?P<lead_time>\d{3})\.nc'

    pressure_level_dict = {'1000': 0, '975': 1, '950': 2, '925': 3, '900': 4, '875': 5, '850': 6, '800': 7,
                           '750': 8, '700': 9, '650': 10, '600': 11, '550': 12, '500': 13, '450': 14, '400': 15,
                           '350': 16, '300': 17, '250': 18, '200': 19, '150': 20, '100': 21}

    default_variables = [
        'T:850', 'T:700', 'T:500',
        'rh_liq:850', 'rh_liq:700', 'rh_liq:500',
        'rain', 'q2m', 'rh2m', 't2m', 'tsfc', 'ps'
    ]

    def __init__(self, root_dir: str, variable_filter: str = None):
        if variable_filter is None:
            variable_filter = ','.join(self.default_variables)
        super().__init__(root_dir, variable_filter)

    def load_array(self, origin: datetime, lead_time: int, return_variables=False):
        feature_maps = []
        variables_used = []
        variables = self.variable_filter.split(',')
        variables = [v.strip() for v in variables]

        # Load dfp, dfs dataset files
        dfp_dataset = self._load_dfp_dataset(origin, lead_time)
        dfs_dataset = self._load_dfs_dataset(origin, lead_time)
        previous_dfs_dataset = None
        if lead_time > 0:  # load previous dfs_dataset to adjust for accumulated variables
            # TODO: optimize redundant file reads
            previous_dfs_dataset = self._load_dfs_dataset(origin, lead_time - 1)

        # Retrieve feature maps based on selected variables
        for v in variables:
            if ':' in v:
                name, pressure = v.split(":")
                pressure_level = self.pressure_level_dict[pressure]
                if name == 'uv':
                    u = dfp_dataset['u'][0][pressure_level]
                    v = dfp_dataset['v'][0][pressure_level]
                    ws = np.sqrt(u * u + v * v)
                    wd = 180 * np.arctan2(v, u) / np.pi
                    wd = np.where(wd >= 0, wd, wd + 360)
                    wd = np.where(wd < 270, 270 - wd, 270 - wd + 360)
                    feature_maps.append(ws)
                    feature_maps.append(wd)
                    variables_used.append('ws:{}'.format(pressure))
                    variables_used.append('wd:{}'.format(pressure))
                elif name in ['u', 'v', 'T', 'rh_liq', 'hgt']:
                    feature_maps.append(dfp_dataset[name][0][pressure_level])
                    variables_used.append(v)
                else:
                    raise Exception('Invalid variable: {}'.format(v))
            else:
                if v == 'rain':
                    if previous_dfs_dataset is None:
                        rainc = dfs_dataset['rainc_acc'][0]
                        rainl = dfs_dataset['rainl_acc'][0]
                    else:
                        rainc = dfs_dataset['rainc_acc'][0] - previous_dfs_dataset['rainc_acc'][0]
                        rainl = dfs_dataset['rainl_acc'][0] - previous_dfs_dataset['rainl_acc'][0]
                    feature_maps.append(rainc + rainl)
                    variables_used.append('rain')
                elif v == 'snow':
                    if previous_dfs_dataset is None:
                        snowc = dfs_dataset['snowc_acc'][0]
                        snowl = dfs_dataset['snowl_acc'][0]
                    else:
                        snowc = dfs_dataset['snowc_acc'][0] - previous_dfs_dataset['snowc_acc'][0]
                        snowl = dfs_dataset['snowl_acc'][0] - previous_dfs_dataset['snowl_acc'][0]
                    feature_maps.append(snowc + snowl)
                    variables_used.append('snow')
                elif v in ['hpbl', 'pbltype', 'psl', 'tsfc', 'topo', 'ps']:
                    feature_maps.append(dfs_dataset[v][0])
                    variables_used.append(v)
                elif v in ['q2m', 'rh2m', 't2m', 'u10m', 'v10m']:
                    feature_maps.append(dfs_dataset[v][0][0])
                    variables_used.append(v)
                else:
                    raise Exception('Invalid variable: {}'.format(v))

        dfp_dataset.close()
        dfs_dataset.close()
        previous_dfs_dataset.close() if previous_dfs_dataset else None

        # Unmask array (parity w/ previous implementation)
        array = np.stack(feature_maps, axis=0)
        _array = array.filled()
        # assert array.mask is False

        if return_variables:
            return _array, variables_used
        else:
            return _array

    def _verify_data_path_dict(self, data_path_dict):
        if len(data_path_dict) == 0:
            raise Exception('No data found (regardless of dates)')

        for (origin_time, lead_time), paths in data_path_dict.items():
            if len(paths) != 2:
                raise Exception('Invalid data paths found (there should be exactly two files): {}'.format(paths))

    def _load_dfp_dataset(self, origin, lead_time) -> nc.Dataset:
        paths = self.data_path_dict[(origin, lead_time)]
        paths = list(filter(lambda p: 'dfp' in p, paths))
        if len(paths) != 1:
            raise AssertionError(
                'Multiple or no dfp filepath for timestamp {}: \n{}'.format((origin, lead_time), paths))
        return nc.Dataset(paths[0])

    def _load_dfs_dataset(self, origin, lead_time) -> nc.Dataset:
        paths = self.data_path_dict[(origin, lead_time)]
        paths = list(filter(lambda p: 'dfs' in p, paths))
        if len(paths) != 1:
            raise AssertionError(
                'Multiple or no dfs filepath for timestamp {}: \n{}'.format((origin, lead_time), paths))
        return nc.Dataset(paths[0])


class GdapsUmBaseDataset(BaseDataset):
    """
    GDPS_UM_NPY 저장 형태 -> (변수, 데이터) shape = (253, 2)
    인덱스로 접근해서 변수를 불러오는 형식입니다.
    default_variables외의 변수를 사용하고자 할 때 변수명에 해당하는 인덱스(변수번호 - 1)를 확인해야합니다.
    
    현재 사용 변수
    Temperature (85000, 70000, 50000) - 변수번호 (140, 143,147)
    Relative humidity (85000, 70000, 50000) - 변수번호 (207, 210, 214)
    rain - 변수번호 (1, 15) 두 변수 합쳐서 사용 (누적되는 지 확인 필요)
    Specific humidity (1.5m) - 변수번호 (6)
    Relative humidity (1.5m) - 변수번호 (7)
    Temperature (1.5m) - 변수번호 (5)
    Temperature (surface) - 변수번호 (248)
    Pressure (surface) - 변수번호 (253)
    """
    dataset_dir = 'GDPS_UM_NPY_FW'
    data_path_glob = os.path.join('**', '*.npy')
    data_path_regex = '.*gdps_um_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})_f(?P<lead_time>\d{3}).*.npy'

    default_variables = [139, 142, 146, 206, 209, 213, 0, 14, 6, 4, 252, 18]

    def __init__(self, root_dir: str, variable_filter: str = None):
        if variable_filter is None:
            variable_filter = self.default_variables
        super().__init__(root_dir, variable_filter)

    def load_array(self, origin: datetime, lead_time: int, return_variables=False):
        feature_maps = []
        variables_used = []
        variables = self.variable_filter

        # current_rain = self.get_feature_array(origin, lead_time, 0).transpose() + self.get_feature_array(origin, lead_time, 14).transpose()
        current_rain = self.get_feature_array(origin, lead_time, 0) + self.get_feature_array(origin, lead_time, 14)
        feature_maps.append(current_rain)
        variables_used.extend([0, 14])

        # Retrieve feature maps based on selected variables
        for v in variables:
            if v >= 0 and v <= 252:
                if v == 0 or v == 14:
                    continue
                # feature_maps.append(self.get_feature_array(origin, lead_time, v).transpose())
                feature_maps.append(self.get_feature_array(origin, lead_time, v))
                variables_used.append(v)
            else:
                raise Exception('Invalid variable: {}'.format(v))

        array = np.stack(feature_maps, axis=0).transpose([0, 2, 1])
        if return_variables:
            return array, variables_used
        else:
            return array

    def get_feature_array(self, origin: datetime, lead_time: int, index: int):
        paths = self.data_path_dict[(origin, lead_time)]
        tag = 'feature_{:03d}'.format(index)
        for path in paths:
            if tag in path:
                return np.load(path)
        # If not found
        raise ValueError(
            'Could not find UM data from origin={}, lead_time={}, feature={}'.format(origin, lead_time, index))

    def _verify_data_path_dict(self, data_path_dict):
        if len(data_path_dict) == 0:
            raise Exception('No data found (regardless of dates)')


class AwsBaseDataset(BaseDataset):
    dataset_dir = None  # NWP model must be specified in subclass
    data_path_glob = os.path.join('**', '*.npy')
    data_path_regex = '.*AWS_HOUR_ALL_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<lead_time>\d{2})_\d*.npy'

    def __init__(self, root_dir: str, variable_filter: str = None):
        if variable_filter is not None:
            print('Warning: `variable_filter` argument is not used in `AWSBaseDataset`')
        if self.dataset_dir is None:
            raise NotImplementedError('You must specify the attribute `dataset_dir`')
        super().__init__(root_dir, variable_filter)

    def load_array(self, origin: datetime, lead_time: int, return_variables=False):
        path = self.data_path_dict[(origin, lead_time)][0]  # only one path per timestamp
        gt = np.load(path).astype(np.float32)
        gt[gt < 0] = -9999.0
        if return_variables:
            return gt, ['observed_rain']
        else:
            return gt

    def _verify_data_path_dict(self, data_path_dict):
        if len(data_path_dict) == 0:
            raise Exception('No data found (regardless of dates)')

        for (origin_time, lead_time), paths in data_path_dict.items():
            if len(paths) != 1:
                raise Exception(
                    'Invalid data paths found (there should be exactly one file for each timestamp): {}'.format(
                        paths))


class AwsBaseDatasetForGdapsKim(AwsBaseDataset):
    dataset_dir = 'AWS_GDPS_KIM_GRID'


class AwsBaseDatasetForGdapsUm(AwsBaseDataset):
    dataset_dir = 'AWS_GDPS_UM_GRID'


def print_dataset_info(dataset: BaseDataset):
    print('Data Paths (subset):')
    print('-' * 80)
    for (origin, lead_time), paths in list(dataset.data_path_dict.items())[:3]:
        print('Timestamp: {} (lead_time={}):'.format(origin, lead_time))
        for path in paths:
            print('\t{}'.format(path))
    print('=' * 80)
    print('Data Format:')
    print('-' * 80)
    array, variables = dataset.load_array(datetime(2020, 7, 1, 0), 0, return_variables=True)
    print('Array shape: {}'.format(array.shape))
    print('Variables used: {}'.format(variables))
    print('=' * 80)


def main():
    root_dir = '/home/osilab12/ssd4'
    variable_filter = 'rain, hpbl, pbltype, psl, tsfc, topo, q2m, T:850, T:700, T:500, uv:850, uv:700, uv:500'
    dataset = GdapsKimBaseDataset(root_dir, variable_filter)
    print('GdapsKimBaseDataset Test'.center(40).center(80, '='))
    print_dataset_info(dataset)

    print()
    print()
    print()
    print()
    print()

    root_dir = '/home/osilab12/ssd4'
    variable_filter = [139, 142, 146, 206, 209, 213, 0, 14, 5, 6, 4, 247, 252]
    dataset = GdapsUmBaseDataset(root_dir, variable_filter)
    print('GdapsUmBaseDataset Test'.center(40).center(80, '='))
    print_dataset_info(dataset)

    print()
    print()
    print()
    print()
    print()

    root_dir = '/home/osilab12/ssd4'
    dataset = AwsBaseDatasetForGdapsKim(root_dir)
    print('AwsBaseDatasetForGdapsKim Test'.center(40).center(80, '='))
    print_dataset_info(dataset)


if __name__ == '__main__':
    main()
