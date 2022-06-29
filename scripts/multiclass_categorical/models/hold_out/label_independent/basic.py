import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import pickle
import sys

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/multiclass/processed/CICDDoS_pre.csv", index_col=[0])

X = df.drop(columns=[' Label'])
y = df[' Label']

input_features = len(X.columns)
train_size = float(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

start = time.time()
# Feature Selection
fs_start = time.time()
threshold = float(sys.argv[2])
selector = VarianceThreshold(threshold=threshold)
selector.fit(X_train)

features_to_keep = X.columns[selector.get_support()]

X_train = selector.transform(X_train)
X_train = pd.DataFrame(X_train)
X_train.columns = features_to_keep

X_test = selector.transform(X_test)
X_test = pd.DataFrame(X_test)
X_test.columns = features_to_keep

X_train.drop(columns=' Fwd Header Length.1', inplace=True)
X_test.drop(columns=' Fwd Header Length.1', inplace=True)
fs_end = time.time()
# Training the model
train_start = time.time()
clf_xgb = xgb.XGBClassifier(eval_metric="mlogloss", seed=42)
# clf_xgb.fit(X_train, 
#             y_train,
#             # verbose=True,
#             ## the next three arguments set up early stopping.
#             early_stopping_rounds=5,
#             eval_metric='logloss',
#             eval_set=[(X_test, y_test)])
clf_xgb.fit(X_train, y_train)
train_end = time.time()
end = time.time()

fs_time = (fs_end - fs_start)
train_time = (train_end - train_start)
fit_time = (end - start)

output_features = len(X_train.columns)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      xticks_rotation='vertical')
plt.savefig(root + 'pictures/multiclass_categorical/label_independent/confusion_matrix_basic.pdf', bbox_inches='tight')

y_pred = clf_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score(y_test, y_pred, average='macro')

pfm = pd.DataFrame(["Basic_Methods", accuracy, precision, recall, f1score,
                    fit_time, fs_time, train_time, input_features, output_features])
pfm = pfm.T
pfm.columns = ["Method", "Accuracy", "Precision", "Recall", "F1_Score", 
                "Fit_Time", "FS_Time", "Train_Time", "Input_Features", "Output_Features"]

filename = root + "pickles/multiclass_categorical/hold_out/label_independent/basic.pkl"
outfile = open(filename, 'wb')
pickle.dump(pfm, outfile)
outfile.close()