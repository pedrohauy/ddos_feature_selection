import os

files = ["./cross_validation_binary.py",
        "./cross_validation_multiclass_categorical.py",
        "./cross_validation_multiclass_one_vs_one.py",
        "./cross_validation_multiclass_one_vs_rest.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)