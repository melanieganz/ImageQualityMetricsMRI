#!/bin/bash

# activate conda env:
source /home/iml/hannah.eichhorn/anaconda3/bin/activate
conda activate dev_iqm

# store current date in variable
now=$(date +"%y_%m_%d")

cd /home/iml/hannah.eichhorn/Code/_OWN_PUBLIC_REPS/ImageQualityMetricsMRI/

nohup python -u process_all_subjects.py --normalisation "percentile" --mask_metric_values True --reduction "worst" --apply_brainmask True > Results/log_"$now"_baseline.txt &
wait

nohup python -u process_all_subjects.py --normalisation "percentile" --mask_metric_values True --reduction "mean" --apply_brainmask True > Results/log_"$now"_reduction-mean.txt &
wait

nohup python -u process_all_subjects.py --normalisation "percentile" --mask_metric_values False --reduction "worst" --apply_brainmask True > Results/log_"$now"_mask-mulitply.txt &
wait

nohup python -u process_all_subjects.py --normalisation "percentile" --mask_metric_values False --reduction "worst" --apply_brainmask False > Results/log_"$now"_mask-none.txt &
wait

nohup python -u process_all_subjects.py --normalisation "min_max" --mask_metric_values True --reduction "worst" --apply_brainmask True > Results/log_"$now"_norm-min-max.txt &
wait

nohup python -u process_all_subjects.py --normalisation "mean_std" --mask_metric_values True --reduction "worst" --apply_brainmask True > Results/log_"$now"_norm-mean_std.txt &
wait

nohup python -u process_all_subjects.py --normalisation "none" --mask_metric_values True --reduction "worst" --apply_brainmask True > Results/log_"$now"_none.txt &
wait
