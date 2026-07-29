#!/bin/bash
# usage: run_one.sh <gpu> <cfg_name> <main.py args...>
set -u
REPO=/mindopt/caoyuan/global_searching/code_repos/diffusion-tts
PY=/home/mindopt/.conda/envs/diffusion-tts/bin/python
GPU=$1; CFG=$2; shift 2
DIR=$REPO/results/day0/$CFG
mkdir -p "$DIR"; cd "$DIR"
echo "[queue] start $CFG gpu=$GPU $(date '+%F %T')" >> $REPO/results/day0/queue_gpu$GPU.log
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/main.py "$@" > run.log 2>&1
rc=$?
echo "[queue] done  $CFG gpu=$GPU rc=$rc $(date '+%F %T')" >> $REPO/results/day0/queue_gpu$GPU.log
