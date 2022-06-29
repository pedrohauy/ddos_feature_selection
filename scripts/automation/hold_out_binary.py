import os
from tqdm import tqdm

root = "../../"

bin_ind = "scripts/binary/models/hold_out/label_independent/"
bin_dep = "scripts/binary/models/hold_out/label_dependent/"

parameters = {
    "train_size": "0.9",
    "variance_threshold": "0.01",
    "correlation_threshold": "0.9",
    "dep_output_features": "20"
}

files = [
    root + bin_ind + "pre.py" + " " + parameters["train_size"],
    root + bin_ind + "basic.py" + " " + parameters["train_size"] + " " + parameters["variance_threshold"],
    root + bin_ind + "corr.py" + " " + parameters["train_size"] + " " + parameters["correlation_threshold"],

    root + bin_dep + "anova.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"],
    root + bin_dep + "mutual_info.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"],
    root + bin_dep + "relieff.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"],
    root + bin_dep + "xgb_gain.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"],
    root + bin_dep + "ensemble_wrfs.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"],
    root + bin_dep + "rfe.py" + " " + parameters["train_size"] + " " + parameters["dep_output_features"]
]

for file in tqdm(files, desc="Hold-Out Binary Models"):
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)