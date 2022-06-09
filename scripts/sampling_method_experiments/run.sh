#!/bin/bash

GPU_INDICES=(0 1 2 3)

export CUDA_VISIBLE_DEVICES=$GPU_INDEX

MODELS=("unet" "convlstm" "metnet")

GPU_INDEX=${GPU_INDICES[INDEX % 4]}
METHOD=$(( INDEX / 3))
MODEL=${MODELS[INDEX % 3]}

if [[ $METHOD == 0 ]]; then
  python train.py --model=$MODEL --device="$GPU_INDEX" --input_data="gdaps_kim" \
    --reference=aws --num_epochs=20 --normalization \
    --start_lead_time 6 --end_lead_time 88 \
    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
    --rain_thresholds 0.1 10.0 \
    --interpolate_aws \
    --intermediate_test \
    --custom_name="method_vanilla"

elif [[ $METHOD == 1 ]]; then
  python train.py --model=$MODEL --device="$GPU_INDEX" --input_data="gdaps_kim" \
    --reference=aws --num_epochs=20 --normalization \
    --start_lead_time 6 --end_lead_time 88 \
    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
    --rain_thresholds 0.1 10.0 \
    --interpolate_aws \
    --intermediate_test \
    --dry_sampling_rate=0.2 \
    --custom_name="method_undersample_0.20"

elif [[ $METHOD == 2 ]]; then
  python train.py --model=$MODEL --device="$GPU_INDEX" --input_data="gdaps_kim" \
    --reference=aws --num_epochs=20 --normalization \
    --start_lead_time 6 --end_lead_time 88 \
    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
    --rain_thresholds 0.1 10.0 \
    --interpolate_aws \
    --intermediate_test \
    --no_rain_ratio=0.2 \
    --custom_name="method_balance_0.20"

fi
