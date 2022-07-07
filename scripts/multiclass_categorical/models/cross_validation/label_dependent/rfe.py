import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import pickle
import sys
from sklearn.feature_selection import RFE
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

root = "../../../../../"
df = pd.read_csv(root + "datasets/multiclass/processed/CICDDoS_corr.csv", index_col=[0])

performance = []

max_features = int(sys.argv[2]) + 1

for features in tqdm(range(1,max_features), desc="RFE Feature Selection"):
    X = df.drop(columns=[' Label'])
    y = df[' Label']

    splits = 10
    fit_times = []
    predict_times = []
    test_sizes = []
    score_times = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train,  X_test = X.iloc[train_index], X.iloc[test_index]
        y_train,  y_test = y.iloc[train_index], y.iloc[test_index]

        start = time.time()
        # Feature Selection
        clf_xgb_rfe = xgb.XGBClassifier(verbosity=0, seed=42, booster='gbtree')
        selector = RFE(clf_xgb_rfe, n_features_to_select=features, step=1)
        selector = selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        # Training the model
        clf_xgb = xgb.XGBClassifier(verbosity=0, seed=42)
        # clf_xgb.fit(X_train, 
        #             y_train,
        #             # verbose=True,
        #             # the next three arguments set up early stopping.
        #             early_stopping_rounds=5,
        #             eval_metric='logloss',
        #             eval_set=[(X_test, y_test)])
        clf_xgb.fit(X_train, y_train)
        end = time.time()
        fit_times.append(end - start)

        start = time.time()
        y_pred = clf_xgb.predict(X_test)
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

    fit_times = np.array(fit_times)
    predict_times = np.array(predict_times)
    test_sizes = np.array(test_sizes)
    test_accuracies = np.array(test_accuracies)
    test_precisions = np.array(test_precisions)
    test_recalls = np.array(test_recalls)
    test_f1_scores = np.array(test_f1_scores)
    score_times = np.array(score_times)

    pfm = pd.DataFrame([test_accuracies, test_precisions, test_recalls, test_f1_scores,
                        fit_times, predict_times, score_times, test_sizes])
    pfm = pfm.T
    pfm.columns = ["Accuracy", "Precision", "Recall", "F1_Score", 
                    "Fit_Time", "Predict_Time", "Score_Time", "Test_Size"]
    performance.append(pfm)

filename = root + "pickles/multiclass_categorical/cross_validation/label_dependent/rfe.pkl"
outfile = open(filename, 'wb')
pickle.dump(performance, outfile)
outfile.close()