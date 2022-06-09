#!/bin/bash

VARS=(
	"u:500, u:700, u:850, v:500, v:700, v:850, T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, tsfc, ps"
	"rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rain, pbltype, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, hgt:500, hgt:700, hgt:850, rain, pbltype, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, hpbl, pbltype, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, psl, q2m, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, rh2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, t2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, tsfc, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, tsfc, u10m, v10m, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, tsfc, topo, ps"
	"T:500, T:700, T:850, rh_liq:500, rh_liq:700, rh_liq:850, rain, pbltype, q2m, rh2m, t2m, tsfc"
)

VAR_KEYS=(
	"u_v_T_rh_liq_rain_pbltype_q2m_rh2m_t2m_tsfc_ps"
	"rh_liq_rain_pbltype_q2m_rh2m_t2m_tsfc_ps"
	"T_rain_pbltype_q2m_rh2m_t2m_tsfc_ps"
	"T_rh_liq_hgt_rain_pbltype_q2m_rh2m_t2m_tsfc_ps"
	"T_rh_liq_rain_hpbl_pbltype_q2m_rh2m_t2m_tsfc_ps"
	"T_rh_liq_rain_q2m_rh2m_t2m_tsfc_ps"
	"T_rh_liq_rain_pbltype_psl_q2m_rh2m_t2m_tsfc_ps"
	"T_rh_liq_rain_pbltype_rh2m_t2m_tsfc_ps"
	"T_rh_liq_rain_pbltype_q2m_t2m_tsfc_ps"
	"T_rh_liq_rain_pbltype_q2m_rh2m_tsfc_ps"
	"T_rh_liq_rain_pbltype_q2m_rh2m_t2m_ps"
	"T_rh_liq_rain_pbltype_q2m_rh2m_t2m_tsfc_u10m_v10m_ps"
	"T_rh_liq_rain_pbltype_q2m_rh2m_t2m_tsfc_topo_ps"
	"T_rh_liq_rain_pbltype_q2m_rh2m_t2m_tsfc"
)

GPU_INDICES=(0 1 2 3 4 5 6)
GPU_INDEX=${GPU_INDICES[INDEX]}

export CUDA_VISIBLE_DEVICES=$GPU_INDEX

for I in $(seq $((INDEX * 2)) $((INDEX * 2 + 1))); do
  for MODEL in "unet" "convlstm" "metnet"; do
    VAR=${VARS[I]};
    VAR_KEY=${VAR_KEYS[I]};
    python train.py --model=$MODEL --device="$GPU_INDEX" --seed=0 --input_data="gdaps_kim" \
      --reference=aws --num_epochs=20 --normalization \
      --start_lead_time 6 --end_lead_time 88 \
      --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
      --variable_filter "$VAR" \
      --rain_thresholds 0.1 10.0 \
      --interpolate_aws \
      --intermediate_test \
      --custom_name="var_$VAR_KEY"
  done
done
