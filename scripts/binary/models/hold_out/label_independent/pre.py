import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import pickle
import sys

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/binary/processed/CICDDoS_pre.csv", index_col=[0])
df[' Label'] = df[' Label'].apply(lambda x: 'ATTACK' if x != 'BENIGN' else 'BENIGN')

encoding = {
    "BENIGN": 0,
    "ATTACK" : 1    
}
df[' Label'] = df[' Label'].map(encoding)

X = df.drop(columns=[' Label'])
y = df[' Label']

input_features = 86
train_size = float(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

start = time.time()
clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", seed=42)
# clf_xgb.fit(X_train, 
#             y_train,
#             verbose=True,
#             ## the next three arguments set up early stopping.
#             early_stopping_rounds=5,
#             eval_metric='logloss',
#             eval_set=[(X_test, y_test)])
clf_xgb.fit(X_train, y_train)
end = time.time()

fs_time = 0
train_time = (end - start)
fit_time = (end - start)

output_features = len(X_train.columns)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      display_labels=encoding.keys(),
                      values_format='d',
                      xticks_rotation='vertical')
plt.savefig(root + 'pictures/binary/label_independent/confusion_matrix_pre.pdf', bbox_inches='tight')

y_pred = clf_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

pfm = pd.DataFrame(["Preprocessing", accuracy, precision, recall, f1score,
                    fit_time, fs_time, train_time, input_features, output_features])
pfm = pfm.T
pfm.columns = ["Method", "Accuracy", "Precision", "Recall", "F1_Score", 
                "Fit_Time", "FS_Time", "Train_Time", "Input_Features", "Output_Features"]

filename = root + "pickles/binary/hold_out/label_independent/pre.pkl"
outfile = open(filename, 'wb')
pickle.dump(pfm, outfile)
outfile.close()