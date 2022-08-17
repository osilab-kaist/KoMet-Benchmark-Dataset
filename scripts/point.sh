python train.py --model="point" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=100 --normalization \
                --rain_thresholds 0.1 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --start_dim 12 \
                --wd 0. \
                --num_workers 0 \
                --custom_name="point_custom_model_spatial"