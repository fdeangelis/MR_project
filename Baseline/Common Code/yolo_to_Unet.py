import os
import numpy as np
import cv2

'''This script converts YOLO-format polygon segmentation annotations (.txt text files with normalized coordinates) 
into 2D pixel-level semantic segmentation mask images (.png), 
typically used to prepare datasets for architectures like U-Net.'''

def read_labels(label_path):
    label = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  
            cls_id = int(parts[1]) 
            coords = list(map(float, parts[2:]))  
            
            label.append([cls_id] + coords)
    return label

def yolo_seg_to_mask(img, label_path):
    label = read_labels(label_path)
    
    H, W = img.shape[:2]
    
    mask = np.full((H, W), 21, dtype=np.int32)
    
    num_classes = len(label)

    for idx, coords in enumerate(label):
        cls_id = coords[0]  
        points = []
        for i in range(1, len(coords), 2):  
            x = int(coords[i] * W)  
            y = int(coords[i+1] * H)  
            
            points.append((x, y))
        
        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(mask, [points], color=cls_id)  

    return mask, num_classes

def generate_masks_for_dataset(imgs_path, labels_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    
    img_files = os.listdir(imgs_path)

    i=0
    
    for img_file in img_files:
        img_path = os.path.join(imgs_path, img_file)
        label_path = os.path.join(labels_path, img_file.replace(".png", ".txt"))  

        img = cv2.imread(img_path)
        
        
        mask, num_classes = yolo_seg_to_mask(img, label_path)
        
        mask_output_path = os.path.join(output_dir, img_file.replace(".png", "_mask.png"))
        cv2.imwrite(mask_output_path, mask.astype(np.uint8))  
        i+=1
        print(f"Processing: {i}/12216")

imgs_path = "../Dataset/dataset/images/train"
labels_path = "../Dataset/dataset/labels/train"

output_dir = "../Dataset/dataset_uNet/masks/train"

generate_masks_for_dataset(imgs_path, labels_path, output_dir)
