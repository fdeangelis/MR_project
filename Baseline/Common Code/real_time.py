import torch
import cv2
from transformers import pipeline
from Model import ROIClassifier
from pipeline import *
from pipeline_with_depth_treshold import *
from PIL import  Image
import time
import matplotlib.pyplot as plt

def to_tti_name(name):
    if name == None:
      return 'unknown_tti'
    name_to_id = {
        12 : 'unknown_tti',
        13: 'coagulation',
        14: 'other',
        15: 'retract and grab',
        16: 'blunt dissection',
        17: 'energy - sharp dissection',
        18: 'staple',
        19: 'retract and push',
        29: 'cut - sharp dissection',
    }

    return name_to_id[name]


def real(video, yolo_model, depth, tti_classifier, device):
    vidcap = cv2.VideoCapture(video)

    
    success, image = vidcap.read()
    H_full, W_full = image.shape[:2]

    # VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter('processed_video.mp4', fourcc,15.0, (W_full, H_full))
    
    while success:
        before = time.time()
        detection , tti_predictions  = end_to_end_pipeline(image, yolo_model, depth, tti_classifier, device)
        after = time.time()
        print("TIME: ",after-before)
        
        overlay_mask = None    
        combined_tool_mask = np.zeros((H_full, W_full), dtype=np.uint8)
        combined_tissue_mask = np.zeros((H_full, W_full), dtype=np.uint8)
        box_mask = None
        
        # Aggiungi una variabile per disegnare le bounding boxes separatamente
        bounding_boxes = []
        # print(tti_predictions)
        for elem in tti_predictions:
            tool_mask = elem['tool']['mask']
            tissue_mask = elem['tissue']['mask']
            is_tti = elem['tti_class']
            
            tool_class = elem['tool']['class']
            tissue_class = elem['tissue']['class']

            tool_mask = tool_mask.astype(np.uint8)
            
            tool_mask_resized = cv2.resize(
                tool_mask.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            )
            tissue_mask_resized = cv2.resize(
                tissue_mask.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            )
            
            combined_tool_mask = np.maximum(combined_tool_mask, tool_mask_resized)
            combined_tissue_mask = np.maximum(combined_tissue_mask, tissue_mask_resized)

            overlay_image_tool = show_mask_overlay_from_binary_mask(image, combined_tool_mask, alpha=0.5, mask_color=(0.0, 0.0, 1.0))
            overlay_image_tissue = show_mask_overlay_from_binary_mask(image, combined_tissue_mask, alpha=0.5, mask_color=(0.0, 1.0, 0.0))
            overlay_mask = cv2.bitwise_or(overlay_image_tool, overlay_image_tissue)

            if is_tti == 1:
                box_mask = (tool_mask_resized + tissue_mask_resized).clip(0, 1).astype('uint8') 
                x, y, w, h = cv2.boundingRect(box_mask)
                
                bounding_boxes.append((x, y, w, h))
            
        
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(overlay_mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #Added the following line
            cv2.putText(overlay_mask, f"Class: {to_tti_name(int(tissue_class))}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
        
        if overlay_mask is not None:
            # cv2.imshow("video", overlay_mask)
            # plt.imshow(cv2.cvtColor(overlay_mask,cv2.COLOR_BGR2RGB))
            # plt.show()
            # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            # plt.show()
            out.write(overlay_mask)
        else:
            # plt.imshow(image)
            # plt.show()
            # cv2.imshow("video", image)
            out.write(image)
            

        # cv2.waitKey(1)
      
        success, image = vidcap.read()
        if image is None:
            break
  

    vidcap.release()
    cv2.destroyAllWindows()
    out.release()

        


if __name__ == "__main__":
    model = load_yolo_model('./runs_OLD_DATASET/segment/train/weights/best.pt')
    video = '../Dataset/video_dataset/videos/test/Adnanset-Lc 78-001.mp4'

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tti_class = ROIClassifier(2)
    tti_class.load_state_dict(torch.load('ROImodel.pt',map_location=device))
    tti_class.to(device)
    
    real(video,model,pipe,tti_class,device)
    
    
    
    