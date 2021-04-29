import os
import pandas as pd
import numpy as np

root = '../../../'

frames = [pd.read_csv(root + 'datasets/multiclass/samples/' + file, index_col=[0])
             for file in os.listdir(root + 'datasets/multiclass/samples/') if file.endswith(".csv")]

df_cicddos = pd.concat(frames)
df_cicddos.to_pickle(root + "pickles/general/multiclass_balanced_labels.pkl")

columns_to_remove = ['Flow ID', ' Source IP', ' Source Port',
       ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp', 'SimillarHTTP', ' Inbound']
df_no_categorical = df_cicddos.drop(columns=columns_to_remove)

integer_columns = df_no_categorical.select_dtypes(include="int64").columns
df_no_categorical[integer_columns] = df_no_categorical[integer_columns].astype("int32")

float_columns = df_no_categorical.select_dtypes(include="float64").columns
df_no_categorical[float_columns] = df_no_categorical[float_columns].astype("float32")

df_optimized = df_no_categorical.drop(columns=[' Label']).copy()

df_optimized.replace([np.inf, -np.inf], np.nan, inplace=True)
df_optimized.fillna(df_optimized.max(), inplace=True)

negatives = df_optimized.columns[(df_optimized < 0).any()]
df_optimized[negatives] = df_optimized[negatives].abs()

df_optimized[' Label'] = df_no_categorical[' Label']
df_optimized.to_csv(root + 'datasets/multiclass/processed/CICDDoS_pre.csv')
