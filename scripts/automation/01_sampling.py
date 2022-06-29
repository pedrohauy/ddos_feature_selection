import os

files = ["../binary/processing/sampling.py",
        "../multiclass_categorical/processing/sampling.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)