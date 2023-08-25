import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from lama.model import Lama
#import lama_cleaner 
#from lama_cleaner.model import lama

import sys
print(sys.path)


def inpaint_with_lama(model, org_img_path, mask_path, output_path):
    org_img = cv2.imread(org_img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = (mask > 0.5).astype(np.float32)

    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    org_img_tensor = transform(org_img).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        inpainted_img = model(org_img_tensor, mask_tensor)

    inpainted_img = inpainted_img.squeeze().cpu().numpy()
    inpainted_img = np.transpose(inpainted_img, (1, 2, 0))

    cv2.imwrite(output_path, inpainted_img * 255.0)

def main():
    data_directory = "data"
    original_directory = os.path.join(data_directory, "Original")
    mask_directory = os.path.join(data_directory, "Mask")
    output_parent_directory = os.path.join(data_directory, "InpaintedImages")

    model = lama.from_pretrained('lama-m')

    for image_filename in os.listdir(original_directory):
        if image_filename.lower().endswith((".jpg", ".png")):
            org_img_path = os.path.join(original_directory, image_filename)
            base_name, ext = os.path.splitext(image_filename)

            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_directory, mask_filename)

            output_folder = os.path.join(output_parent_directory, base_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_filename = f"{base_name}_inpainted{ext}"
            output_path = os.path.join(output_folder, output_filename)

            inpaint_with_lama(model, org_img_path, mask_path, output_path)

if __name__ == "__main__":
    main()
