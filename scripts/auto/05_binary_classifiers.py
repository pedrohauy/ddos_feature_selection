import os

files = ["../binary/models/cross_validation/anova.py",
        "../binary/models/cross_validation/mutual_info.py",
        "../binary/models/cross_validation/relieff.py",
        "../binary/models/cross_validation/xgb_gain.py",
        "../binary/models/cross_validation/ensemble_wrfs.py",
        "../binary/models/cross_validation/RFE.py"]

#files = ["../binary/models/cross_validation/anova.py"]

print("\n" + "#"*50)
print("\tTraining Binary Classifiers")
print("#"*50 + "\n")

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)