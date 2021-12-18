import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/facial-keypoints-detection/training/training.csv')
    print("Number of columns: ", len(df.columns))
    # Check the column-wise distribution of null values

    res = len(df.columns) - 1 - df.isnull().sum(axis=1)
    print(res.value_counts())

    # Output is:
    # Number of columns:  31
    # 8     4755
    # 30    2140
    # 28      87
    # 26      28
    # 6       10
    # 24       9
    # 22       8
    # 18       5
    # 20       3
    # 16       2
    # 10       2
