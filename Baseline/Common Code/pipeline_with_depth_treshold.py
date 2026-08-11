import torch
from transformers import pipeline
from PIL import Image
from pipeline import *
import cv2
import numpy as np

def expand_mask(mask, pixels):
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded



def depth_treshold(image, yolo_model, depth_model):
    # Step 1: YOLOv11-seg and depth estimation
  
    detections = yolo_inference(yolo_model, image)

    # depth_map = depth_model(image)
    depth_map = np.array(depth_model(image)["depth"])
    


    # depth_map = np.array(depth_model(Image.fromarray(image))["depth"])

    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)

    image = cv2.imread(image,cv2.IMREAD_COLOR)


    
    tti_predictions = []

    for pair in pairs:
        tool_mask = pair['tool']['mask']
        tissue_mask = pair['tissue']['mask']
        roi = extract_union_roi(image, tool_mask, tissue_mask, depth_map)
        if roi is None:
            return [] ,[]
        

        H_full, W_full = depth_map.shape[:2]

        tool_mask_expanded = expand_mask(tool_mask, pixels=5)
        tissue_mask_expanded = expand_mask(tissue_mask, pixels=5)

        tool_mask_resized = cv2.resize(
                tool_mask_expanded.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        tissue_mask_resized = cv2.resize(
            tissue_mask_expanded.astype(np.uint8),
            (W_full, H_full),
            interpolation=cv2.INTER_NEAREST
            ).astype(bool)




        intersection = np.logical_and(tool_mask_resized, tissue_mask_resized).sum()

        tti = False

        if intersection > 0:

            # depth_tool = np.where(tool_mask_resized, depth_map, 0)
            # depth_tissue = np.where(tissue_mask_resized, depth_map, 0)

            # cv2.imshow("Tool Mask", depth_tool.astype(np.uint8) * 255)
            # cv2.imshow("Tissue Mask", depth_tissue.astype(np.uint8) * 255)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            intersection_pixels = np.logical_and(tool_mask_resized, tissue_mask_resized)

            im = np.where(intersection_pixels, depth_map, 0)

            # cv2.imshow("Intersection", im.astype(np.uint8) * 255)
            # cv2.waitKey(0)

            depth_intersection = depth_map[intersection_pixels]


            max_depth = np.max(depth_intersection)
            

            depth_intersection = depth_intersection/max_depth
            max_depth = np.max(depth_intersection)
            min_depth = np.min(depth_intersection)

            

            if max_depth - min_depth <= 1:
                # print("TTI")
                tti = True


        if tti:
            tti_class = 1
        else:
            tti_class = 0
            
        # Save ROI result
        tti_predictions.append({
            'tool': pair['tool'],
            'tissue': pair['tissue'],
            'tti_class': tti_class,
            'tti_score': 1
        })

    return detections, tti_predictions



if __name__ == "__main__":
    model = load_yolo_model('./runs_OLD_DATASET/segment/train/weights/best.pt')
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth_treshold("../Dataset/evaluation/evaluation/images/video0004_frame0068.png", model, depth, device)