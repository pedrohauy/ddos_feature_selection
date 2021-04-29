import pandas as pd
import os
from tqdm import tqdm

sample_size = int((12 * 22 * 20)/12)

root = '../../../'

# subtract 1 because of TFTP_shrinked.csv
n_files = len(os.listdir(root + 'datasets/base/')) - 1

for file in tqdm(os.listdir(root + 'datasets/base/'), desc="Sampling multiclass"):
    if file.endswith(".csv"):
        if file != "TFTP.csv": # ignore TFTP_shrinked.csv
            df = pd.read_csv(root + 'datasets/base/' + file, index_col=[0])

            labels = []
            for label in df[' Label'].unique():
                if label != 'WebDDoS':
                    df_filtered = df[df[' Label'] == label]
                    if label != 'BENIGN':
                        df_label = df_filtered.sample(n=sample_size, replace=True, random_state=42)
                    else:
                        df_label = df_filtered.sample(n=round(sample_size/n_files),
                                                        replace=True, random_state=42)
                    labels.append(df_label)

            df_sampled = pd.concat(labels)
            sampled_filename = root + 'datasets/multiclass/samples/' + file.split('.')[0] + '_sampled.csv'
            df_sampled.to_csv(sampled_filename)