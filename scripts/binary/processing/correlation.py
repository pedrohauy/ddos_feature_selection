import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = '../../../'
df = pd.read_csv(root + 'datasets/binary/processed/CICDDoS_basic.csv', index_col=[0])
X = df.drop(columns=[' Label'])

corrmat = X.corr()

sns.set(rc={'figure.figsize':(20,15)})
ax = sns.heatmap(
    corrmat, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=100)
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
)
plt.savefig(root + 'pictures/general/binary_correlation.png', bbox_inches='tight')

corrmat = corrmat.abs().unstack() # absolute value of corr coef
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat[corrmat > 0.9]
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

df_no_correlation = X.drop(columns=features_to_remove)
df_no_correlation[' Label'] = df[' Label']

df_no_correlation.to_csv(root + "datasets/binary/processed/CICDDoS_corr.csv")