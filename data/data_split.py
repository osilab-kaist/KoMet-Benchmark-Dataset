from datetime import timedelta, datetime
from typing import List

from torch.utils.data import Subset

from data.dataset import StandardDataset


def interval_split(dataset: StandardDataset, date_intervals):
    split_indices: List[List[int]] = [list() for _ in date_intervals]  # list of empty lists
    for i, target_timestamps in enumerate(dataset.target_timestamps):
        target_origin = target_timestamps[0]
        for j, (start, end) in date_intervals:
            if start <= target_origin < end + timedelta(days=1):
                split_indices[j].append(i)
                break
        else:
            raise AssertionError('Something is wrong with this code')

    split_datasets = []
    for indices in split_indices:
        split_datasets.append(Subset(dataset, indices))
    return split_datasets


def cyclic_split(dataset: StandardDataset, split_days=(4, 2, 2), cycle_start_delta=0):
    cycle_start_day = datetime(year=1970, month=1, day=1) + timedelta(days=cycle_start_delta)
    cycle_length = sum(split_days)
    split_indices = [list() for _ in split_days]  # list of empty lists

    for i, (origin, _) in enumerate(dataset.target_timestamps):
        delta = (origin - cycle_start_day).days
        delta %= cycle_length
        cutoff = 0
        for j, days in enumerate(split_days):
            cutoff += days
            if delta < cutoff:
                split_indices[j].append(i)
                break
        else:
            raise AssertionError('Something is wrong with this code')

    split_datasets = []
    for indices in split_indices:
        split_datasets.append(Subset(dataset, indices))

    return split_datasets


def main():
    """
    Ad-hoc test code for `cyclic_split`
    """

    class MockDataset(StandardDataset):
        def __init__(self):
            pass

        def __getitem__(self, index):
            return self.target_timestamps[index]

    dataset = MockDataset()
    dataset.target_timestamps = []

    for d in range(100):
        for h in range(24):
            dt = datetime(year=1970, month=1, day=1) + timedelta(days=d)
            dataset.target_timestamps.append((dt, h))
    train, val, test = cyclic_split(dataset)
    for ds in [train, val, test]:
        timestamps = []
        for t in ds:
            timestamps.append(t)
        print(timestamps[:10])


if __name__ == "__main__":
    main()
