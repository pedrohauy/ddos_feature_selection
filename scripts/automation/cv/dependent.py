import os

files = ["./05_binary_classifiers.py",
        "./06_multiclass_categorical_classifiers.py",
        "./07_multiclass_one_vs_one_classifiers.py",
        "./08_multiclass_one_vs_rest_classifiers.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)