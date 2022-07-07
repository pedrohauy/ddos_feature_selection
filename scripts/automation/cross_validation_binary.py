import os

root = "../../"

bin_ind = "scripts/binary/models/cross_validation/label_independent/"
bin_dep = "scripts/binary/models/cross_validation/label_dependent/"

parameters = {
    "splits": "10",
    "variance_threshold": "0.01",
    "correlation_threshold": "0.9",
    "dep_max_features": "40"
}

files = [
    root + bin_ind + "pre.py" + " " + parameters["splits"],
    root + bin_ind + "basic.py" + " " + parameters["splits"] + " " + parameters["variance_threshold"],
    root + bin_ind + "corr.py" + " " + parameters["splits"] + " " + parameters["correlation_threshold"],

    root + bin_dep + "anova.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + bin_dep + "mutual_info.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + bin_dep + "relieff.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + bin_dep + "xgb_gain.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + bin_dep + "ensemble_wrfs.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + bin_dep + "rfe.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"]
]

print("\n" + "#"*50)
print("\tTraining Binary Classifiers")
print("#"*50 + "\n")

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)