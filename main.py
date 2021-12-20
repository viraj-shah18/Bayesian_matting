import yaml
import os
import cv2
import matplotlib.pyplot as plt
from matting import bayesian_matting


def load_imgs(img_name, IMAGE_FOLDER, TRIMAP_FOLDER, GT_FOLDER):
    # reading the images and converting BGR image to RGB. Change image name for any other image
    trimap1_path = os.path.join(TRIMAP_FOLDER, "Trimap1", img_name)
    trimap2_path = os.path.join(TRIMAP_FOLDER, "Trimap2", img_name)
    IMAGE_PATH = os.path.join(IMAGE_FOLDER, img_name)
    trimap1 = cv2.imread(trimap1_path, cv2.IMREAD_GRAYSCALE)
    trimap2 = cv2.imread(trimap2_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.imread(IMAGE_PATH)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    gt_path = os.path.join(GT_FOLDER, img_name)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = gt / 255
    return input_image, trimap1, trimap2, gt


def save_imgs(img, trimap, pred, gt, save_name):
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.title("Combined Trimap")
    plt.imshow(trimap, cmap="gray", vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.title("Predicted Alpha Map")
    plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.title("Actual Alpha Map")
    plt.imshow(gt, cmap="gray", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_name)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    IMAGE_NAME = config['img_name']
    IMAGE_FOLDER = config['img_folder']
    TRIMAP_FOLDER = config['trimap_folder']
    GT_FOLDER = config['gt_folder']
    SAVE_FOLDER = config['output_folder']
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    save_name = config['output_name']
    save_name = os.path.join(SAVE_FOLDER, save_name)

    image, trimap1, trimap2, gt = load_imgs(IMAGE_NAME, IMAGE_FOLDER, TRIMAP_FOLDER, GT_FOLDER)
    pixels_to_consider = 35
    combined_trimap, alpha_map = bayesian_matting(
        image, trimap1, trimap2, pixels_to_consider
    )
    save_imgs(image, combined_trimap, alpha_map, gt, save_name)
