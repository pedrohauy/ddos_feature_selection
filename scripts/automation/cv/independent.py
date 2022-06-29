import os

files = ["./02_preprocessing_classifiers.py",
        "./03_basic_classifiers.py",
        "./04_correlation_classifiers.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)