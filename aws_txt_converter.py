"""
Module to convert AWS observations of `.txt` format to numpy format based on coordinate information in
`aws_coordinates/`.
"""
import argparse
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytz
from dateutil import tz
from tqdm import tqdm

from utils import parse_date

re_fname = re.compile("AWS_HOUR_ALL_(\d{4})(\d{2})(\d{2})(\d{2})00")


def get_station_coordinate(codi_aws_df, stn_id):
    stn_info = codi_aws_df[codi_aws_df['stn'] == stn_id]
    x = stn_info['xii'] - 1
    y = stn_info['yii'] - 1

    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDAPS Observations Converter')
    parser.add_argument('--dataset_dir', default='/home/osilab12/ssd4', type=str, help='root directory of dataset')
    parser.add_argument('--input_data', default='gdaps_kim', type=str, help='input data: gdaps_kim, gdaps_um, ldaps')
    parser.add_argument('--start_date', default='2020-07', type=str,
                        help='start date in YYYY-MM or YYYY-MM-DD form, inclusive (KST)')
    parser.add_argument('--end_date', default='2020-07',
                        help='end date in YYYY-MM or YYYY-MM-DD form (end of month for format 1), inclusive (KST)')
    parser.add_argument('--realtime', action="store_true",
                        help='generate time based on current time KST')
    args = parser.parse_args()

    if args.realtime:
        today = datetime.now(pytz.timezone('Asia/Seoul')).date()
        args.start_date = today.strftime("%Y-%m-%d")
        args.end_date = today.strftime("%Y-%m-%d")
        print("Realtime argument enabled. Using dates {} - {}".format(args.start_date, args.end_date))

    if args.input_data == 'gdaps_kim':
        codi_aws_df = pd.read_csv("aws_coordinates/codi_gdps_kim_aws.csv")

    start_date = parse_date(str(args.start_date), end=False)
    end_date = parse_date(str(args.end_date), end=True)

    root_dir = args.dataset_dir

    KST = tz.gettz('Asia/Seoul')
    obs_txt_dir = os.path.join(root_dir, 'AWS', str(start_date.year))
    if not os.path.isdir(obs_txt_dir):
        raise Exception("AWS directory for year {} does not exist".format(start_date.year))

    # List of AWS observation txt file path basenames
    obs_txt_list = sorted([f for f in os.listdir(obs_txt_dir) if
                           f.split('_')[3][:-8] >= '{}'.format(start_date.strftime("%Y%m%d")) and
                           f.split('_')[3][:-8] <= '{}'.format(end_date.strftime("%Y%m%d"))])

    # Process each observation file (corresponding to a single target timestamp)
    pbar = tqdm(obs_txt_list)
    for obs_txt in pbar:
        # Get target path
        # Convert KST time to UTC time
        match = re_fname.search(obs_txt)
        if not match:
            raise AssertionError('Invalid AWS txt path: ', obs_txt)
        y, m, d, h = match.groups()
        kst_date = datetime(int(y), int(m), int(d), hour=int(h), tzinfo=KST)
        utc_date = kst_date.astimezone(tz=timezone.utc)
        utc_year, utc_month, utc_day, utc_hour, utc_minute = \
            utc_date.year, utc_date.month, utc_date.day, utc_date.hour, utc_date.minute
        utc_str = '{:4d}{:02d}{:02d}{:02d}{:02d}'.format(utc_year, utc_month, utc_day,
                                                         utc_hour, utc_minute)
        file_name = 'AWS_HOUR_ALL_{}_{}.npy'.format(utc_str, utc_str)
        obs_npy_dir = os.path.join(root_dir, 'AWS_{}_GRID'.format(args.input_data.upper()), str(start_date.year))
        os.makedirs(obs_npy_dir, exist_ok=True)
        target_path = os.path.join(obs_npy_dir, file_name)

        # Skip file if already generated
        if os.path.isfile(target_path):
            continue

        pbar.set_description('[current_file] {}'.format(obs_txt))

        if args.input_data == 'gdaps_kim':
            result_array = np.full([50, 65], -9999., dtype=np.float32)

        # Parse observation values and copy to numpy array
        with open(os.path.join(obs_txt_dir, obs_txt), 'r', encoding='euc-kr') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                # Get proper rain data and station coordinate
                line_list = line.strip().split()
                stn_id = int(line_list[1])
                one_hour_rain = float(line_list[6])
                stn_x, stn_y = get_station_coordinate(codi_aws_df, stn_id)
                result_array[stn_y, stn_x] = one_hour_rain

        # Save observations to `.npy` file
        with open(target_path, 'wb') as npyf:
            np.save(npyf, result_array)
