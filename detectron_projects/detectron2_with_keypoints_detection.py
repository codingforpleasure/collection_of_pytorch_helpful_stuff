# Attention For getting a proper csv file as input,
# first run the script: 'rotate_keypoints_with_opencv_for_generating_synthetic_data.py'

import glob
import shutil
import numpy as np
import pandas as pd
import cv2  # was installed from the whl file
import os, json, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import matplotlib.colors as mcolors
from random import sample
from collections import Iterable
from detectron2.checkpoint import DetectionCheckpointer


def get_matches_dicts(img_dir, dict_class_to_number):
    dataset_dicts = []
    idx = 0

    df = pd.read_csv('matches/train/keypoints_with_background_img.csv')
    groups_by_filename = df.groupby("filename")

    # removing the size height and width
    # Excluding height,weight,
    num_keypoints = int((len(list(df.columns)) - 5) / 2)

    # keypoints_padding = np.array(keypoints_padding)
    # keypoints_padding[2::3] = 2

    for img_filename_path in glob.glob(os.path.join(img_dir, "*.jpeg")):
        height, width = cv2.imread(img_filename_path).shape[:2]
        img_filename = os.path.basename(img_filename_path)

        all_keyframes_for_all_objects_on_page = groups_by_filename.get_group(img_filename)
        all_keyframes_for_all_objects_on_page.drop('filename', 1, inplace=True)

        record = dict()

        record["file_name"] = img_filename_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        idx += 1
        objs = []

        for index, row_per_object in all_keyframes_for_all_objects_on_page.iterrows():
            row_per_object = np.array(row_per_object)

            # Regarding the keypoints:
            # in the format of [x1, y1, v1,…, xn, yn, vn].
            # v[i] means the visibility of this keypoint.
            # n must be equal to the number of keypoint categories.
            # The Xs and Ys are absolute real-value coordinates in range [0, W or H].

            actual_keypoints = row_per_object[:-4]  # remove bbox info (last four columns)
            keypoints_with_visibilty = np.insert(actual_keypoints, list(range(2, num_keypoints * 2 + 2, 2)), 2)
            keypoints_with_visibilty = list(keypoints_with_visibilty)

            # bbox-height,bbox-width,bbox-uppercorner-x,bbox-uppercorner-y
            bbox_h, bbox_w, bbox_x1, bbox_y1 = row_per_object[-4:]

            obj = {"bbox": [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h],
                   "bbox_mode": BoxMode.XYXY_ABS,
                   "category_id": 1,
                   "iscrowd": 0,
                   "num_keypoints": num_keypoints,
                   "keypoints": keypoints_with_visibilty}
            # obj["keypoint_flip_map"] = [("head", "head")]
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


class Trainer(DefaultTrainer):
    # If you use DefaultTrainer, you can overwrite its build_{train,test}_loader method to use your own dataloader.
    # See the deeplab dataloader for an example.
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        print("Gilc 'build_test_loader' was invoked!!!!")
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        print("Gilc 'build_train_loader' was invoked!!!!")
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                      augmentations=[
                                                                          T.RandomFlip(prob=0.5,
                                                                                       horizontal=True),
                                                                          T.RandomFlip(prob=0.5,
                                                                                       horizontal=False,
                                                                                       vertical=True)
                                                                      ])
                                            # ,
                                            # T.RandomRotation(angle=[0, 180]),
                                            )


if __name__ == '__main__':

    # df = pd.read_csv(
    #     "/home/gil_diy/PycharmProjects/detectron_2022_clean_start/matches/train/keypoints_with_background_img.csv")
    #
    # img = cv2.imread("/home/gil_diy/PycharmProjects/detectron_2022_clean_start/matches/train/match3_keypoint_rotated_with_background17.jpeg")
    # cv2.rectangle(img,
    #               pt1=((df["bbox-uppercorner-x"].values)[0], (df["bbox-uppercorner-y"].values)[0]),
    #               pt2=((df["bbox-uppercorner-x"].values)[0] + (df["bbox-width"].values)[0],
    #                    (df["bbox-uppercorner-y"].values)[0] + (df["bbox-height"].values)[0]),
    #               color=(255, 0, 0),
    #               thickness=2)
    #
    # cv2.imshow("show", img)
    # cv2.waitKey(0)

    classes_names = ['junk', 'match']
    num_classes = len(classes_names)
    colors_palette = [(255, 241, 0)]
    # , (255, 140, 0), (232, 17, 35),
    # (236, 0, 140), (104, 33, 122), (0, 24, 143),
    # (0, 188, 242), (0, 178, 148), (0, 158, 73),
    # (186, 216, 10)
    classes_values = list(range(len(classes_names)))

    dict_class_name_to_index = dict()
    for class_shape, index_class in zip(classes_names, classes_values):
        dict_class_name_to_index[class_shape] = index_class

    colors_dict = dict()
    colors = mcolors.TABLEAU_COLORS.keys()
    for class_shape, color in zip(classes_names, colors):
        colors_dict[class_shape] = color

    for d in ["train", "val"]:
        # the name that identifies a dataset
        DatasetCatalog.register(name="matches/" + d, func=lambda d=d: get_matches_dicts("matches/" + d,
                                                                                        dict_class_name_to_index))

        MetadataCatalog.get(name="matches/" + d).set(thing_classes=classes_names)
        # not sure??
        # MetadataCatalog.get(name="matches/" + d).set(thing_colors=colors_palette)
        MetadataCatalog.get(name="matches/" + d).set(keypoint_names=["head", "head"])
        MetadataCatalog.get(name="matches/" + d).set(keypoint_flip_map=[])

    shapes_metadata = MetadataCatalog.get("matches/" + d)

    dataset_dicts = get_matches_dicts("matches/train", dict_class_name_to_index)

    for d in sample(dataset_dicts, 3):
        print("The filename is: ", ["file_name"])
        img = cv2.imread(d["file_name"])

        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=shapes_metadata,
                                scale=1,
                                instance_mode=ColorMode.SEGMENTATION)

        out = visualizer.draw_dataset_dict(d)

        # for box in outputs["instances"].pred_boxes.to('cpu'):
        #     visualizer.draw_box(box)
        #     visualizer.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))

        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey()

    cv2.destroyAllWindows()

    #################################### Configuration ####################################

    cfg = get_cfg()
    res = model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # keypoint_rcnn_R_50_FPN_3x.yaml
    # res = model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # keypoint_rcnn_R_50_FPN_3x.yaml
    cfg.merge_from_file(res)  #
    cfg.DATASETS.TRAIN = ("matches/train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to val longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = os.path.join('matches/output')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold

    trainer = Trainer(cfg)
    print("1) trainer")
    model = build_model(cfg)
    print("2) build_model")
    trainer.resume_or_load(resume=False)
    print("3) trainer.resume_or_load")

    trainer.train()

    full_model_path = os.path.join(cfg.OUTPUT_DIR, "model_matches_keypoints.pth")
    shutil.move(src=os.path.join(cfg.OUTPUT_DIR, "model_matches_keypoints.pth"),
                dst=full_model_path)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_matches_keypoints.pth")  # path to the model we just trained

    ################################################## Inference ##################################################

    predictor = DefaultPredictor(cfg)

    dir_img = "/home/gil_diy/PycharmProjects/detectron_2022_clean_start/matches"
    imgs_path = glob.glob(os.path.join(dir_img, "*.jpeg"))
    for idx, img_path in enumerate(imgs_path):
        im = cv2.imread(img_path)
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=shapes_metadata,
                       scale=1,
                       instance_mode=ColorMode.SEGMENTATION
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imshow("image: " + img_path, out.get_image()[:, :, ::-1])
        cv2.waitKey()
        cv2.destroyWindow("image: " + img_path)
