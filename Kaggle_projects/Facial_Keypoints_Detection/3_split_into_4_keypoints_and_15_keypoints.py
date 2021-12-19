import shutil
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('/media/gil_diy/DEECBE15ECBDE843/Datasets_kaggle/facial-keypoints-detection/training/training.csv')

    # The meaning of df.isnull().sum(axis=1) == 0 is all 30 columns are full -> 15 keypoints (x,y)
    consist_15_keypoints = np.where(df.isnull().sum(axis=1) == 0)

    # The meaning of df.isnull().sum(axis=1) == 4 is all 8 columns are full -> 4 keypoints (x,y)
    consist_4_keypoints = np.where(df.isnull().sum(axis=1) == 22)
    print("consist_15_keypoints : ", consist_15_keypoints, "quantity: ", len(consist_15_keypoints[0]))
    print("consist_4_keypoints : ", consist_4_keypoints, "quantity: ", len(consist_4_keypoints[0]))

    # The rest are leftover so we discarded them

    ######### Moving the images to the corresponding directories `keypoints_4` and `keypoints_15`

    # for img_idx in consist_15_keypoints:
    #     shutil.move(src=f'training/face_{img_idx}.jpeg',
    #                 dst=f'training/keypoints_15/face_{img_idx}.jpeg')

    # for img_idx in consist_4_keypoints:
    #     shutil.move(src=f'training/face_{img_idx}.jpeg',
    #                 dst=f'training/keypoints_4/face_{img_idx}.jpeg')

    ######### Creating two csv files`train_15_keypoints.csv` and `train_4_keypoints.csv`

    df_15_keypoints = df.iloc[consist_15_keypoints[0], :-1]
    df_15_keypoints['filename'] = consist_15_keypoints[0]
    df_15_keypoints['filename'] = df_15_keypoints['filename'].apply(lambda x: 'face_' + str(x) + '.jpeg')
    df_15_keypoints.to_csv('train_15_keypoints.csv', index=False)

    df_4_keypoints = df.iloc[consist_4_keypoints[0], :-1]
    df_4_keypoints['filename'] = consist_4_keypoints[0]
    df_4_keypoints['filename'] = df_4_keypoints['filename'].apply(lambda x: 'face_' + str(x) + '.jpeg')
    df_4_keypoints.to_csv('train_4_keypoints.csv', index=False)
