import os

files = ["./hold_out_binary.py",
        "./hold_out_multiclass_categorical.py",
        "./hold_out_multiclass_one_vs_one.py",
        "./hold_out_multiclass_one_vs_rest.py"]

for file in files:
    home = os.getcwd()
    os.chdir(os.path.dirname(file))
    os.system("python ./" + os.path.basename(file))
    os.chdir(home)