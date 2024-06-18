"""
Temporary code to load one example dataset for trying out the metrics for now.
"""

# ToDo: proper normalization and masking!
#   - check if Tenengrad, NGS and perceptual metric values make sense after masking

import argparse
import os
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")
from data_utils import load_data
from metrics.fsim import calc_fsim
from metrics.gradient_metrics import *
from metrics.perceptual_metric import perceptual_metric

metrics_dict = {
    "full_reference": {
        "FSIM": calc_fsim,
        "PerceptualMetric": perceptual_metric},
    "reference_free": {
        "Tenengrad": tenengrad,
        "NGS": normalized_gradient_squared,
        "GradientEntropy": gradient_entropy}
}

parser = argparse.ArgumentParser(description="Image Quality Metrics")
parser.add_argument("--config_path",
                    type=str,
                    default="configs/config_debug.yaml",
                    metavar="C",
                    help="path to configuration yaml file")
args = parser.parse_args()
with open(args.config_path, "r") as stream_file:
    config_file = yaml.load(stream_file, Loader=yaml.FullLoader)

subject = config_file["data"]["subjects"][0]

reference = load_data(
    os.path.join(config_file["data"]["folder"], subject, "anat"),
    acq="mpragepmcoff",
    rec="wore",
    run="01"
)
image = load_data(
    os.path.join(config_file["data"]["folder"], subject, "anat"),
    acq="mpragepmcoff",
    rec="wore",
    run="02"
)

reference = reference / np.amax(reference)
image = image / np.amax(image)

for metric in config_file["output"]["metrics"]:
    if metric in metrics_dict["full_reference"]:
        print(f"{metric}: {metrics_dict['full_reference'][metric](image, reference)}")

    elif metric in metrics_dict["reference_free"]:
        print(f"{metric}: {metrics_dict['reference_free'][metric](image)}")

    else:
        raise NotImplementedError("Metric {} not implemented.".format(metric))
