#!/bin/bash

# Determine the relevant directories based on hostname
if [ "$(hostname)" == "PC085765" ]; then
    home_directory="/home/iml/hannah.eichhorn/"
    anaconda_directory="/home/iml/hannah.eichhorn/anaconda3/"
    code_directory="/home/iml/hannah.eichhorn/Code/_OWN_PUBLIC_REPS/ImageQualityMetricsMRI/"
    host_id="bacio"

elif [ "$(hostname)" == "marche06-13mba.wireless.nyumc.org" ]; then
    home_directory="/Users/emarchetto/"
    anaconda_directory="/Users/emarchetto/miniconda3"
    code_directory="/Users/emarchetto/Documents/PROJECTS/ImageQualityMetricsMRI"
    host_id="elisatest"

else
    echo "Unknown hostname: $(hostname)"
    exit 1
fi

# activate conda env:
source $anaconda_directory/bin/activate
conda activate dev_iqm

# change directory
cd $code_directory



