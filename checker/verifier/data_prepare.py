"""
    Creating NOTENOUGHINFO data by replacing evidence with random sampled evidence
"""

import os
import log
import random
import argparse
import pandas as pd
from tqdm import tqdm

def main(args):
    random.seed(args.random_state)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file_name in ['train.csv', 'dev.csv']:
        data = []
        evidence = []
        df = pd.read_csv(os.path.join(args.input_dir, file_name))
        for _, instance in df.iterrows():
            evidence.append(instance['evidence'])

        nei_rows = []
        idx_list = list(range(len(df)))

        for idx, row in tqdm(df.iterrows(), total = len(df), desc = f"[Info] Processing {file_name}"):
            nei_row = row.copy()
            nei_row['label'] = "NOT ENOUGH INFO"

            sampled_idx = idx

            while sampled_idx == idx:
                sampled_idx = random.choice(idx_list)

            nei_row['evidence'] = evidence[sampled_idx]

            nei_rows.append(nei_row)

        df_nei = pd.DataFrame(nei_rows)
        df_final = pd.concat([df, df_nei], ignore_index = True)
        df_final.to_csv(os.path.join(args.output_dir, file_name), index = False, encoding = 'utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Prepare NOTENOUGHINFO labeled sample for training verifier")
    parser.add_argument('--input_dir', type = str)
    parser.add_argument('--output_dir', type = str)
    parser.add_argument('--random_state', default = 16, type = int)
    args = parser.parse_args()

    main(args)