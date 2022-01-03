# This script was written for a subsequent script for generating synthetic data
# It was written in generic style so it would suit to other datasets.

# The output csv format would be:
# filename,keypoint1-x,keypoint1-y,keypoint2-x,keypoint2-y, keypoint3-x,keypoint3-y,...,width,height

import glob
import json
import os
import numpy as np
import pandas as pd
import json


def convert_rec(x):
    if isinstance(x, list):
        return list(map(convert_rec, x))
    else:
        return int(x)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    num_keypoints = 2
    keypoints_header = []
    with open(
            '/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/custom_keypoints_with_torch_from_medium_matches/generated_matches/new_annotations/check2.json',
            'r') as fd:
        data = json.load(fd)
        list_of_points_zeros = []
        list_of_points_ones = []
        bboxes_per_img = []
        for shape in data["shapes"]:
            if shape["label"] == '0':
                values = list(map(lambda x: int(x), shape["points"][0]))
                list_of_points_zeros.append(values + [1])
            elif shape["label"] == '1':
                values = list(map(lambda x: int(x), shape["points"][0]))
                list_of_points_ones.append(values + [1])
            elif shape["label"] == "rec":
                # TODO: make sure (x1,y1) is smaller than  (x2,y2) - otherwise more BUGS wil Arrive!!!
                values = list(np.array(shape["points"], dtype=np.uint32).flatten())
                if values[0] > values[2]:
                    print(np.array(shape["points"]))
                    break
                values = list(map(lambda x: int(x), values))
                bboxes_per_img.append(values)

                # bboxes_per_img = convert_rec(bboxes_per_img)
                # columns_of_visible = np.array([1] * len(list_of_points_zeros))
                # list_of_points_keypoint0 = np.column_stack([list_of_points_zeros, columns_of_visible]).astype(int)
                # list_of_points_keypoint1 = np.column_stack([list_of_points_ones, columns_of_visible]).astype(int)

        list_of_pairs_of_keypoints = []
        for row_kp1, row_kp2 in zip(list_of_points_zeros, list_of_points_ones):
            two_keypoints = [row_kp1, row_kp2]
            list_of_pairs_of_keypoints.append(two_keypoints)

        # bboxes_per_img_dumped = json.dumps(bboxes_per_img, cls=NumpyEncoder)
        # # Writing the json file per image
        json_body = {
            "bboxes": bboxes_per_img,
            "keypoints": list_of_pairs_of_keypoints
        }

        with open(
                '/home/gil_diy/PycharmProjects/pytorch_interview_preparation_dec_2021/custom_keypoints_with_torch_from_medium_matches/generated_matches/new_annotations/check2_correct.json',
                'w') as fd:
            json.dump(json_body, fd, indent=2, ensure_ascii=False)
        print("gfhg")
        # for idx in range(num_keypoints):
        #     keypoints_header += [f'keypoint{idx}-x', f'keypoint{idx}-y']
        # output_df_columns_names = ['filename'] + keypoints_header + ['width', 'height']
        # json_annotation_input_directory = 'imgs_single_object'
        # json_files = glob.glob(os.path.join(json_annotation_input_directory, '*.json'))
        # all_rows = []
        #
        # for json_file in json_files:
        #     with open(json_file, 'r') as fd:
        #         json_data = json.load(fd)
        #         xy = [shape['points'][0] for shape in json_data['shapes']]
        #         xy_list_per_object = np.round(np.array(xy).reshape(-1),0)
        #         width, height = json_data['imageWidth'], json_data['imageHeight']
        #         row_to_add = [json_data['imagePath']] + list(xy_list_per_object) + [width, height]
        #         all_rows.append(row_to_add)
