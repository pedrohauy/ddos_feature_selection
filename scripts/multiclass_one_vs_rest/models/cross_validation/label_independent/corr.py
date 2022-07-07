import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import time
import pickle
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/multiclass/processed/CICDDoS_basic.csv", index_col=[0])

X = df.drop(columns=[' Label'])
y = df[' Label']

splits = int(sys.argv[1])
fs_times = []
train_times = []
fit_times = []
number_features = []
predict_times = []
test_sizes = []
score_times = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []

skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

for train_index, test_index in tqdm(skf.split(X, y), desc="Correlation Feature Selection", total=splits):
    X_train,  X_test = X.iloc[train_index], X.iloc[test_index]
    y_train,  y_test = y.iloc[train_index], y.iloc[test_index]

    start = time.time()
    # Feature Selection
    fs_start = time.time()
    corrmat = X_train.corr()
    corrmat = corrmat.abs().unstack() # absolute value of corr coef
    corrmat = corrmat.sort_values(ascending=False)
    correlation_threshold = float(sys.argv[2])
    corrmat = corrmat[corrmat > correlation_threshold]
    corrmat = corrmat[corrmat < 1]
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']

    grouped_feature_ls = []
    correlated_groups = []

    for feature in corrmat.feature1.unique():    
        if feature not in grouped_feature_ls:
            # find all features correlated to a single feature
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(correlated_block.feature2.unique()) + [feature]
            # append the block of features to the list
            correlated_groups.append(correlated_block)

    group_head = []

    for group in correlated_groups:
        group_head.append(group.iloc[0,0])

    features_to_remove = [feature for feature in grouped_feature_ls if feature not in group_head]

    X_train.drop(columns=features_to_remove, inplace=True)
    X_test.drop(columns=features_to_remove, inplace=True)
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

    fs_times.append(fs_end - fs_start)
    train_times.append(train_end - train_start)
    fit_times.append(end - start)

    number_features.append(len(X_train.columns))

    start = time.time()
    y_pred = ovr.predict(X_test)
    end = time.time()
    predict_times.append(end - start)

    test_sizes.append(len(y_pred))

    start = time.time()
    test_accuracies.append(accuracy_score(y_test, y_pred))
    test_precisions.append(precision_score(y_test, y_pred, average="macro"))
    test_recalls.append(recall_score(y_test, y_pred, average='macro'))
    test_f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    end = time.time()
    score_times.append(end - start)

fs_times = np.array(fs_times)
train_times = np.array(train_times)
fit_times = np.array(fit_times)
number_features = np.array(number_features)
predict_times = np.array(predict_times)
test_sizes = np.array(test_sizes)
test_accuracies = np.array(test_accuracies)
test_precisions = np.array(test_precisions)
test_recalls = np.array(test_recalls)
test_f1_scores = np.array(test_f1_scores)
score_times = np.array(score_times)

pfm = pd.DataFrame([test_accuracies, test_precisions, test_recalls, test_f1_scores,
                    fit_times, fs_times, train_times, predict_times, score_times, 
                    number_features, test_sizes])
pfm = pfm.T
pfm.columns = ["Accuracy", "Precision", "Recall", "F1_Score", 
                "Fit_Time", "FS_Time", "Train_Time","Predict_Time", "Score_Time", 
                "Number_Features", "Test_Size"]

filename = root + "pickles/multiclass_one_vs_rest/cross_validation/label_independent/corr.pkl"
outfile = open(filename, 'wb')
pickle.dump(pfm, outfile)
outfile.close()
