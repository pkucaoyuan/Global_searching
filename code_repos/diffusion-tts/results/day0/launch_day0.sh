#!/bin/bash
# Day 0 batch: 8-GPU queues (shares GPUs with the running nowcasting training)
set -u
D=/mindopt/caoyuan/global_searching/code_repos/diffusion-tts/results/day0
REPO=/mindopt/caoyuan/global_searching/code_repos/diffusion-tts
R=$D/run_one.sh
chmod +x $R

SD_COMMON="--backend sd --prompt_csv $REPO/prompts.csv --n_runs 20 --repeat_per_prompt 10 --seed 0"
DB_COMMON="--backend sd --prompt_csv $REPO/drawbench200.csv --n_runs 200 --repeat_per_prompt 3 --seed 0"
ONLINE_SD="--thresh_gain_coef 0.1 --thresh_var_coef 0.8 --N 4 --revert_on_negative"

# GPU0: E1 SD Brightness offline-only 400,500
setsid nohup bash -c "
$R 0 e1_sd_eps1_brightness_400 $SD_COMMON --scorer brightness --method epsilon_1 --total_budget 400 --revert_on_negative
$R 0 e1_sd_eps1_brightness_500 $SD_COMMON --scorer brightness --method epsilon_1 --total_budget 500 --revert_on_negative
" > /dev/null 2>&1 &

# GPU1: E1 SD Compressibility offline-only 400,500
setsid nohup bash -c "
$R 1 e1_sd_eps1_compressibility_400 $SD_COMMON --scorer compressibility --method epsilon_1 --total_budget 400 --revert_on_negative
$R 1 e1_sd_eps1_compressibility_500 $SD_COMMON --scorer compressibility --method epsilon_1 --total_budget 500 --revert_on_negative
" > /dev/null 2>&1 &

# GPU2: E1 EDM offline-only all 6 configs (fast)
setsid nohup bash -c "
for B in 144 180 288; do
  $R 2 e1_edm_eps1_brightness_\$B --backend edm --scorer brightness --method epsilon_1 --num_steps 18 --total_budget \$B --n_runs 20 --revert_on_negative
  $R 2 e1_edm_eps1_compressibility_\$B --backend edm --scorer compressibility --method epsilon_1 --num_steps 18 --total_budget \$B --n_runs 20 --revert_on_negative
done
" > /dev/null 2>&1 &

# GPU3: E6 online dual-signal ablation @SD/400
setsid nohup bash -c "
$R 3 e6_sd_gainonly_brightness_400 $SD_COMMON --scorer brightness --method epsilon_online --total_budget 400 --thresh_gain_coef 0.1 --thresh_var_coef 0 --N 4 --revert_on_negative
$R 3 e6_sd_varonly_brightness_400 $SD_COMMON --scorer brightness --method epsilon_online --total_budget 400 --thresh_gain_coef 0 --thresh_var_coef 0.8 --N 4 --revert_on_negative
$R 3 e6_sd_gainonly_compressibility_400 $SD_COMMON --scorer compressibility --method epsilon_online --total_budget 400 --thresh_gain_coef 0.1 --thresh_var_coef 0 --N 4 --revert_on_negative
$R 3 e6_sd_varonly_compressibility_400 $SD_COMMON --scorer compressibility --method epsilon_online --total_budget 400 --thresh_gain_coef 0 --thresh_var_coef 0.8 --N 4 --revert_on_negative
" > /dev/null 2>&1 &

# GPU4: E4 DrawBench-200 Brightness @400: Uniform -> Offline -> GAINS
setsid nohup bash -c "
$R 4 e4_db_uniform_brightness_400 $DB_COMMON --scorer brightness --method eps_greedy --K 8
$R 4 e4_db_eps1_brightness_400 $DB_COMMON --scorer brightness --method epsilon_1 --total_budget 400 --revert_on_negative
$R 4 e4_db_online_brightness_400 $DB_COMMON --scorer brightness --method epsilon_online --total_budget 400 $ONLINE_SD
" > /dev/null 2>&1 &

# GPU5: E4 DrawBench-200 Compressibility @400
setsid nohup bash -c "
$R 5 e4_db_uniform_compressibility_400 $DB_COMMON --scorer compressibility --method eps_greedy --K 8
$R 5 e4_db_eps1_compressibility_400 $DB_COMMON --scorer compressibility --method epsilon_1 --total_budget 400 --revert_on_negative
$R 5 e4_db_online_compressibility_400 $DB_COMMON --scorer compressibility --method epsilon_online --total_budget 400 $ONLINE_SD
" > /dev/null 2>&1 &

# GPU6: E1 SD Brightness offline-only 800 (longest single config)
setsid nohup bash -c "
$R 6 e1_sd_eps1_brightness_800 $SD_COMMON --scorer brightness --method epsilon_1 --total_budget 800 --revert_on_negative
" > /dev/null 2>&1 &

# GPU7: E1 SD Compressibility offline-only 800
setsid nohup bash -c "
$R 7 e1_sd_eps1_compressibility_800 $SD_COMMON --scorer compressibility --method epsilon_1 --total_budget 800 --revert_on_negative
" > /dev/null 2>&1 &

echo "all queues launched"
