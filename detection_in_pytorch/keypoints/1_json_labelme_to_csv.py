# This script was written for a subsequent script for generating synthetic data
# It was written in generic style so it would suit to other datasets.

# The output csv format would be:
# filename,keypoint1-x,keypoint1-y,keypoint2-x,keypoint2-y, keypoint3-x,keypoint3-y,...,width,height

import glob
import json
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    num_keypoints = 2
    keypoints_header = []
    for idx in range(num_keypoints):
        keypoints_header += [f'keypoint{idx}-x', f'keypoint{idx}-y']
    output_df_columns_names = ['filename'] + keypoints_header + ['width', 'height']
    json_annotation_input_directory = 'imgs_single_object'
    json_files = glob.glob(os.path.join(json_annotation_input_directory, '*.json'))
    all_rows = []

    for json_file in json_files:
        with open(json_file, 'r') as fd:
            json_data = json.load(fd)
            xy = [shape['points'][0] for shape in json_data['shapes']]
            xy_list_per_object = np.round(np.array(xy).reshape(-1),0)
            width, height = json_data['imageWidth'], json_data['imageHeight']
            row_to_add = [json_data['imagePath']] + list(xy_list_per_object) + [width, height]
            all_rows.append(row_to_add)

    df = pd.DataFrame(all_rows, columns=output_df_columns_names)
    df.to_csv('imgs_single_object/single_object_keypoints.csv',index = False)
