import torch
from transformers import pipeline, ViTForImageClassification
from PIL import Image
from Model import ROIClassifier
import matplotlib.pyplot as plt
from pipeline import *
import os
from sklearn.metrics import accuracy_score , f1_score ,precision_score ,recall_score ,confusion_matrix ,balanced_accuracy_score
import pickle
from VIT import ROIClassifierViT

IMAGES_TEST = '../Dataset/evaluation/images/'
LABELS_TEST = '../Dataset/evaluation/labels/'



def generate_predictions(yolo_model):

    images = os.listdir(IMAGES_TEST)
    
    l = len(images)
    i = 1

    wrong_class = 0
    no_pred = 0
    good_class = 0
    good_pred = 0
    
    for img in images:
        print(f"Predicting image {i}/{l}")
        i+=1
        label = img.replace('.png', '.txt')
        # d = {(tool_class , tti_class) : is_tti}
        class_list = []
        with open(LABELS_TEST + label, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                line = line.strip()
      
                tool_class = int(line[2])
                tti_class = int(line[4:])
                is_tti = int(line[0])
         
                if tool_class not in class_list:
                    class_list.append(tool_class)
                if tti_class not in class_list:
                    class_list.append(tti_class)
                
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictions = yolo_model.predict(IMAGES_TEST + img,verbose=False)
        
       
        cl = predictions[0].boxes.cls.cpu().detach().numpy()
       
        for c in cl:
            if c  not in class_list:
                wrong_class += 1
            else:
                good_class += 1
        
        pred_len = len(cl)
        truth_len = len(class_list)
        print(class_list)

        diff = abs(pred_len-truth_len)
        
        if diff == 0:
            good_pred += 1
        
        no_pred += diff

    print("Wrong classes: ",wrong_class)
    print("No preds: ",no_pred)
    print("Good classes: ",good_class)
    print("Preds: ",good_pred)
    




if __name__ == "__main__":
    model = load_yolo_model('./runs_OLD_DATASET/segment/train/weights/best.pt')
    
    generate_predictions(model)

