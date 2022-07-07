import os

root = "../../"

mul_ind = "scripts/multiclass_categorical/models/cross_validation/label_independent/"
mul_dep = "scripts/multiclass_categorical/models/cross_validation/label_dependent/"

parameters = {
    "splits": "10",
    "variance_threshold": "0.01",
    "correlation_threshold": "0.9",
    "dep_max_features": "38"
}

files = [
    root + mul_ind + "pre.py" + " " + parameters["splits"],
    root + mul_ind + "basic.py" + " " + parameters["splits"] + " " + parameters["variance_threshold"],
    root + mul_ind + "corr.py" + " " + parameters["splits"] + " " + parameters["correlation_threshold"],

    root + mul_dep + "anova.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + mul_dep + "mutual_info.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + mul_dep + "relieff.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + mul_dep + "xgb_gain.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + mul_dep + "ensemble_wrfs.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"],
    root + mul_dep + "rfe.py" + " " + parameters["splits"] + " " + parameters["dep_max_features"]
]

print("\n" + "#"*50)
print("\tTraining Multiclass Classifiers")
print("#"*50 + "\n")

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)