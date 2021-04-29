import os

files = ["../binary/processing/preprocessing.py",
        "../multiclass_categorical/processing/preprocessing.py",
        "../binary/processing/basic.py",
        "../multiclass_categorical/processing/basic.py",
        "../binary/processing/correlation.py",
        "../multiclass_categorical/processing/correlation.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)