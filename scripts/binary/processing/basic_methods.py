import pandas as pd
from sklearn.feature_selection import VarianceThreshold

root = '../../../'
df_pre = pd.read_csv(root + 'datasets/binary/processed/CICDDoS_pre.csv', index_col=[0])
df = df_pre.drop(columns=[' Label'])

constant_features = [feature for feature in df.columns if df[feature].std() == 0]
df_no_constant = df.drop(columns=constant_features)

# find features with low variance
sel = VarianceThreshold(threshold=0.01)
sel.fit(df_no_constant)

features_to_keep = df_no_constant.columns[sel.get_support()]
df_no_quasi = sel.transform(df_no_constant)
df_no_quasi = pd.DataFrame(df_no_quasi)
df_no_quasi.columns = features_to_keep

df_no_duplicates = df_no_quasi.drop(columns=' Fwd Header Length.1')

df_pre.reset_index(inplace=True)
df_no_duplicates[' Label'] = df_pre[' Label']

df_no_duplicates.to_csv(root + "datasets/binary/processed/CICDDoS_basic.csv")