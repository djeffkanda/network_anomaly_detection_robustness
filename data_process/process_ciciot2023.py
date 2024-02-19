import argparse

# import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
import utils
import os

# from pyspark.sql import SparkSession
# import pyspark.pandas as pd

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1


def process_df(chunk):
    total_rows = deleted_rows = 0
    total_features = 83
    total_rows += chunk.shape[0]

    colums = list(chunk.columns)
    colums.remove('label')
    # Transforming all non numeric values to NaN
    chunk[colums] = chunk[colums].apply(pd.to_numeric, errors='coerce')

    # Replacing INF values with NaN
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    print('Dropping na')

    # Filtering NaN values
    before_drop = chunk.shape[0]
    chunk.dropna(inplace=True)
    deleted_rows += (before_drop - chunk.shape[0])

    # Drop constant columns
    print('Constant colums ....')
    variances = chunk.var(numeric_only=True)
    constant_columns = variances[variances == 0].index
    before_drop = chunk.shape[0]
    chunk = chunk.drop(constant_columns, axis=1)
    deleted_rows += (before_drop - chunk.shape[0])

    print(constant_columns)
    print(chunk.shape)

    # Drop duplicated columns
    before_drop = chunk.shape[0]
    duplicates = set()
    for i in range(0, len(chunk.columns)):
        col1 = chunk.columns[i]
        for j in range(i + 1, len(chunk.columns)):
            col2 = chunk.columns[j]
            if chunk[col1].equals(chunk[col2]):
                print(f"Column {col1} and {col2} are duplicates")
                duplicates.add(col2)

    print(f"Duplicated columns : {duplicates}")
    chunk.drop(duplicates, axis=1, inplace=True)
    print(chunk.shape)
    deleted_rows += (before_drop - chunk.shape[0])

    # pearson correlation heatmap
    # plt.figure(figsize=(30, 30))
    print("Attributes correlation ....")
    corr = chunk.corr(numeric_only=True)
    # sns.heatmap(corr, annot=True, cmap='RdBu', vmin=-1, vmax=1, square=True)  # annot=True
    # plt.show()

    # Correlated columns

    correlated_col = set()
    is_correlated = [True] * len(corr.columns)
    threshold = 0.95
    for i in range(len(corr.columns)):
        if is_correlated[i]:
            for j in range(i):
                if (corr.iloc[i, j] >= threshold) and (is_correlated[j]):
                    colname = corr.columns[j]
                    is_correlated[j] = False
                    correlated_col.add(colname)
    print(correlated_col)
    print(len(correlated_col))

    # before_drop = chunk.shape[0]
    # deleted_rows += (before_drop - chunk.shape[0])

    before_drop = chunk.shape[0]
    chunk.drop(correlated_col, axis=1, inplace=True)
    print(chunk.shape)
    deleted_rows += (before_drop - chunk.shape[0])

    # pearson correlation heatmap
    # plt.figure(figsize=(30, 30))
    # corr = chunk.corr(numeric_only=True)
    # sns.heatmap(corr, annot=True, cmap='RdBu', vmin=-1, vmax=1, square=True)  # annot=True
    # plt.show()

    # Converting labels to binary values
    chunk['Label_cat'] = chunk['label']
    chunk['label'] = chunk['label'].apply(lambda x: NORMAL_LABEL if x == 'BenignTraffic' else ANORMAL_LABEL)

    stats = {
        "Total Rows": str(total_rows),
        "Total Features": len(chunk.columns),
        "Dropped Rows": str(deleted_rows),
        "Rows after clean": str(total_rows - deleted_rows),
        "Ratio": f"{(deleted_rows / total_rows):1.4f}",
        "Features after clean": str(len(chunk.columns))
    }

    return chunk, stats


def clean_step(path_to_files: str, export_path: str) -> pd.DataFrame:
    chunks = []
    schunks = []
    i = 0

    files_path = [f"{path_to_files}/{f}" for f in os.listdir(path_to_files) if f.endswith('.csv')]

    for f in os.listdir(path_to_files):
        if not f.endswith('.csv'):
            continue

        if i > 3:
            break
        i += 1

        print(f"Cleaning file {f}")
        chunk = pd.read_csv(f"{path_to_files}/{f}")

        # sdf = spark.read.csv(f"{path_to_files}/{f}")
        # schunks.append(sdf)
        chunks.append(chunk)

    df = pd.concat(chunks)
    # sdf =

    # sdf = spark.createDataFrame(df)
    df, stats = process_df(df)

    return df, stats


if __name__ == '__main__':
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders.
    path, export_path, backup, _ = utils.parse_args()

    # 0 - Prepare folder structure
    utils.prepare(export_path)
    path_to_clean = f"{export_path}/{utils.folder_struct['clean_step']}/ciciot2023_clean.csv"
    if os.path.isfile(path_to_clean):
        print("Clean file exists. Skipping cleaning step.")
        df = pd.read_csv(path_to_clean)
    else:
        # 1 - Clean the data (remove invalid rows and columns)
        df, clean_stats = clean_step(path, export_path)
        # Save info about cleaning step
        utils.save_stats(export_path + '/cicids2018_info.csv', clean_stats)

    # 2 - Normalize numerical values and treat categorical values
    print("Saving...")
    df['Label_cat'] = df['Label_cat'].astype('category')
    df.to_parquet(f'{export_path}/{utils.folder_struct["minify_step"]}/ciciot2023.gzip')
