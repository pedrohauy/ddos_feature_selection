import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import pickle
import sys
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/binary/processed/CICDDoS_corr.csv", index_col=[0])
df[' Label'] = df[' Label'].apply(lambda x: 'ATTACK' if x != 'BENIGN' else 'BENIGN')

encoding = {
    "BENIGN": 0,
    "ATTACK" : 1    
}
df[' Label'] = df[' Label'].map(encoding)

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
clf_xgb_gain = xgb.XGBClassifier(verbosity=0, seed=42)
clf_xgb_gain.fit(X_train, y_train)
xgb_gain = clf_xgb_gain.feature_importances_

anova = f_classif(X_train, y_train)[0]

mi = mutual_info_classif(X_train, y_train)

relieff = ReliefF(n_neighbors=20, n_features_to_keep=features)
relieff.fit(X_train.to_numpy(),y_train.to_numpy())
relief = relieff.feature_scores

feature_df = pd.DataFrame(columns=["anova", "mi", "relief", "xgb_gain"])
feature_df["anova"] = anova
feature_df["mi"] = mi
feature_df["relief"] = relief
feature_df["xgb_gain"] = xgb_gain
feature_df.index = X.columns

for method in feature_df.columns:
    feature_df[method + '_rank'] = feature_df[method].rank(method='first')

ranks = [column for column in feature_df.columns if 'rank' in column]
feature_df['wrfs'] = feature_df[ranks].sum(axis=1)
feature_df.sort_values(by='wrfs', ascending=False, inplace=True)

features_to_keep = feature_df.index[:features]
X_train = X_train[features_to_keep]
X_test = X_test[features_to_keep]
fs_end = time.time()
# Training the model
train_start = time.time()
clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", seed=42)
# clf_xgb.fit(X_train, 
#             y_train,
#             verbose=False,
#             # the next three arguments set up early stopping.
#             early_stopping_rounds=5,
#             eval_metric='logloss',
#             eval_set=[(X_test, y_test)])
clf_xgb.fit(X_train, y_train)
train_end = time.time()
end = time.time()

fs_time = (fs_end - fs_start)
train_time = (train_end - train_start)
fit_time = (end - start)

output_features = features

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      display_labels=encoding.keys(),
                      values_format='d',
                      xticks_rotation='vertical')
plt.savefig(root + 'pictures/binary/label_dependent/confusion_matrix_wrfs.pdf', bbox_inches='tight')

y_pred = clf_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

pfm = pd.DataFrame(["Ensemble", accuracy, precision, recall, f1score,
                    fit_time, fs_time, train_time, input_features, output_features])
pfm = pfm.T
pfm.columns = ["Method", "Accuracy", "Precision", "Recall", "F1_Score", 
                "Fit_Time", "FS_Time", "Train_Time", "Input_Features", "Output_Features"]

filename = root + "pickles/binary/hold_out/label_dependent/wrfs.pkl"
outfile = open(filename, 'wb')
pickle.dump(pfm, outfile)
outfile.close()