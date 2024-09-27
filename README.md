# ImageQualityMetricsMRI
Repository housing implementations of image quality metrics for MRI data.


## Setting up the Conda environment

1. Clone this repository and navigate to the project directory in your terminal.

3. Create the Conda environment from the `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

4. Activate the Conda environment:

```bash
conda activate dev_iqm
```

5. For installing FSL (to use BET and FLIRT tools for brain mask extraction and registration), follow these instructions: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

6. You're all set! You can now run the scripts in this project.


## Steps to reproduce the analysis

1. Download and preprocess the data with:
```
conda activate dev_iqm
nohup python -u preprocessingData.py > preprocess_data.log &
```

2. Calculate the image quality metrics for all subjects with:
```
conda activate dev_iqm
nohup python -u process_all_subjects.py > Results/log.txt &
```
One can set specific preprocessing settings for `normalisation`, 
`mask_metric_values`, `reduction`, `apply_brainmask` as input arguments. 
`run_process_all_subjects.sh`is a bash script to run different settings sequentially.g

3. Analyse the correlation between the image quality metrics and the observer
scores with the script `correlation_analysis.py`.

4. Compare correlations between different preprocessing settings with the script `compare_preprocessing.py`.
