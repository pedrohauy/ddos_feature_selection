import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import time
import pickle
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/multiclass/processed/CICDDoS_corr.csv", index_col=[0])

X = df.drop(columns=[' Label'])
y = df[' Label']

input_features = len(X.columns)
train_size = float(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

# Number of Features
features = int(sys.argv[2])

start = time.time()
# Feature Selection
fs_start = time.time()
selector = SelectKBest(mutual_info_classif, k=features).fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
fs_end = time.time()
# Training the model
train_start = time.time()
clf_xgb = xgb.XGBClassifier(eval_metric="logloss", seed=42)
# clf_xgb.fit(X_train, 
#             y_train,
#             # verbose=True,
#             ## the next three arguments set up early stopping.
#             early_stopping_rounds=5,
#             eval_metric='logloss',
#             eval_set=[(X_test, y_test)])
ovr = OneVsRestClassifier(clf_xgb)
#clf_xgb.fit(X_train, y_train)
ovr.fit(X_train, y_train)
train_end = time.time()
end = time.time()

fs_time = (fs_end - fs_start)
train_time = (train_end - train_start)
fit_time = (end - start)

output_features = features

plot_confusion_matrix(ovr, 
                      X_test, 
                      y_test,
                      values_format='d',
                      xticks_rotation='vertical')
plt.savefig(root + 'pictures/multiclass_one_vs_rest/label_dependent/confusion_matrix_mutual_info.pdf', bbox_inches='tight')

y_pred = ovr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score(y_test, y_pred, average='macro')

pfm = pd.DataFrame(["Mutual Information", accuracy, precision, recall, f1score,
                    fit_time, fs_time, train_time, input_features, output_features])
pfm = pfm.T
pfm.columns = ["Method", "Accuracy", "Precision", "Recall", "F1_Score", 
                "Fit_Time", "FS_Time", "Train_Time", "Input_Features", "Output_Features"]

filename = root + "pickles/multiclass_one_vs_rest/hold_out/label_dependent/mutual_info.pkl"
outfile = open(filename, 'wb')
pickle.dump(pfm, outfile)
outfile.close()