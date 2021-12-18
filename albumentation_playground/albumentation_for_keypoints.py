import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from colorama import Fore, Style

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=10):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('images/image_for_keypoints.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    keypoints = [
        (355, 349),
        (404, 347),
        (412, 260),
        (338, 266),
        (286, 264),
        (462, 257),
        (293, 153),
        (437, 151),
        (521, 295),
        (234, 305),
        (377, 395),
        (136, 250),
        (361, 227),
        (53, 42),
        (689, 9)
    ]

    vis_keypoints(image, keypoints)

    # Defining a simple augmentation pipeline
    transform = A.Compose(
        [A.HorizontalFlip(p=1)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    transformed = transform(image=image, keypoints=keypoints)
    vis_keypoints(transformed['image'], transformed['keypoints'])

    # Another example of augmentation pipelines

    print(Fore.BLUE + "Let's apply VerticalFlip" + Style.RESET_ALL)

    transform = A.Compose(
        [A.VerticalFlip(p=1)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    transformed = transform(image=image, keypoints=keypoints)
    vis_keypoints(transformed['image'], transformed['keypoints'])

    # Another example of augmentation pipelines

    # We fix the random seed for visualization purposes,
    # so the augmentation will always produce the same result.
    # In a real computer vision pipeline, you shouldn't fix the random
    # seed before applying a transform to the image because,
    # in that case, the pipeline will always output the same image.
    # The purpose of image augmentation is to use different transformations each time

    print(Fore.BLUE + "Let's apply RandomCrop" + Style.RESET_ALL)
    random.seed(7)
    transform = A.Compose(
        [A.RandomCrop(width=200, height=200, p=1)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    transformed = transform(image=image, keypoints=keypoints)
    vis_keypoints(transformed['image'], transformed['keypoints'])

    print(Fore.BLUE + "Let's apply Rotation" + Style.RESET_ALL)
    random.seed(7)
    transform = A.Compose(
        [A.Rotate(p=0.5)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    transformed = transform(image=image, keypoints=keypoints)
    vis_keypoints(transformed['image'], transformed['keypoints'])
