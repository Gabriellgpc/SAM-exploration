from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import cv2

import torch

if __name__ == '__main__':
    image = cv2.imread('/datasets/pexels-kitchen.jpg')
    h, w, _ = image.shape
    image = cv2.resize(image, dsize=[480, int(h/w*480)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["vit_b"](checkpoint="/datasets/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # predictor.set_image(rgb_image)
    # input_prompts =
    # masks, _, _ = predictor.predict(<input_prompts>)

    # or generate masks for an entire image:
    # sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_image)

    print(masks.shape)

    cv2.imshow('image', image)
    cv2.imshow('masks', masks)
    cv2.waitKey(0)