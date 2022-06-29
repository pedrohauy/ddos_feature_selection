import os

files = ["../multiclass_categorical/models/cross_validation/anova.py",
        "../multiclass_categorical/models/cross_validation/mutual_info.py",
        "../multiclass_categorical/models/cross_validation/relieff.py",
        "../multiclass_categorical/models/cross_validation/xgb_gain.py",
        "../multiclass_categorical/models/cross_validation/ensemble_wrfs.py",
        "../multiclass_categorical/models/cross_validation/RFE.py"]

#files = ["../multiclass_categorical/models/cross_validation/ensemble_wrfs.py"]

print("\n" + "#"*50)
print("\tTraining Multiclass Classifiers")
print("#"*50 + "\n")

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)