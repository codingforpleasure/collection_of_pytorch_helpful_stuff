import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    pd_4 = pd.read_csv("train_4_keypoints.csv")
    pd_15 = pd.read_csv("train_15_keypoints.csv")

    train_4, valid_4 = train_test_split(pd_4, test_size=0.2)
    train_15, valid_15 = train_test_split(pd_15, test_size=0.2)

    train_4.to_csv('./training/train_4.csv', index=False)
    valid_4.to_csv('./training/valid_4.csv', index=False)
    train_15.to_csv('./training/train_15.csv', index=False)
    valid_15.to_csv('./training/valid_15.csv', index=False)
    print("Debug")

    directories = ['train_4', 'train_15', 'valid_4', 'valid_15']

    training_dir = '/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/Detecting_Facial_Keypoints/training'

    # for dir in directories:
    #     os.mkdir(os.path.join(training_dir, dir))

    for filename in train_4.iloc[:, -1]:
        print(filename)
        shutil.move(src=os.path.join(training_dir, 'keypoints_4', filename),
                    dst=os.path.join(training_dir, 'train_4'))

    for filename in train_15.iloc[:, -1]:
        print(filename)
        shutil.move(src=os.path.join(training_dir, 'keypoints_15', filename),
                    dst=os.path.join(training_dir, 'train_15'))

    for filename in valid_4.iloc[:, -1]:
        print(filename)
        shutil.move(src=os.path.join(training_dir, 'keypoints_4', filename),
                    dst=os.path.join(training_dir, 'valid_4'))

    for filename in valid_15.iloc[:, -1]:
        print(filename)
        shutil.move(src=os.path.join(training_dir, 'keypoints_15', filename),
                    dst=os.path.join(training_dir, 'valid_15'))
