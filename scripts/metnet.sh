python train.py --model="metnet" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=20 --normalization \
                --rain_thresholds 0.1 1.0 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --auxiliary_loss 0.1 \
                --custom_name="metnet_test"