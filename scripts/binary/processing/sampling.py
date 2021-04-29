import pandas as pd
import os
from tqdm import tqdm

sample_size = int((12 * 22 * 20)/22)

root = '../../../'

for file in tqdm(os.listdir(root + 'datasets/base/'), desc="Sampling binary"):
    if file.endswith(".csv"):
        if file != "TFTP.csv": # ignore TFTP_shrinked.csv
            df = pd.read_csv(root + 'datasets/base/' + file, index_col=[0])

            labels = []
            for label in df[' Label'].unique():
                if label != 'WebDDoS':
                    df_filtered = df[df[' Label'] == label]                    
                    df_label = df_filtered.sample(n=sample_size, replace=True, random_state=42)
                    labels.append(df_label)

            df_sampled = pd.concat(labels)
            sampled_filename = root + 'datasets/binary/samples/' + file.split('.')[0] + '_sampled.csv'
            df_sampled.to_csv(sampled_filename)