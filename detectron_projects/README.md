<!--ts-->
   * [Making dataset ready for detectron](#making-dataset-ready-for-detectron)
      * [Annotation  - CSV format](#annotation----csv-format)
      * [How to create this kind of file format?](#how-to-create-this-kind-of-file-format)
      * [Generating a synthetic dataset](#generating-a-synthetic-dataset)
      * [Straight from labelme](#straight-from-labelme)

<!-- Added by: gil_diy, at: Wed 29 Dec 2021 10:35:02 IST -->

<!--te-->

# Making dataset ready for detectron

## Annotation  - CSV format


filename|keypoint0-x|keypoint0-y|keypoint1-x|keypoint1-y|bbox-width|bbox-height|bbox-uppercorner-x|bbox-uppercorner-y
------------|-----|-----|-----|-----|-----|-----|-----|-----
img1.jpeg|354|84|650|109|45|301|353|74
img1.jpeg|110|138|131|434|301|51|97|137
img3.jpeg|719|31|568|286|265|168|559|27
img3.jpeg|535|297|270|164|150|274|266|155
img3.jpeg|269|165|565|152|26|300|268|147

## How to create this kind of file format?

## Generating a synthetic dataset

This file format CSV can be generated from my script `rotate_keypoints_with_opencv_for_generating_synthetic_data.py`

## Straight from labelme

This file format can be generated from my script `rotate_keypoints_with_opencv_for_generating_synthetic_data.py`