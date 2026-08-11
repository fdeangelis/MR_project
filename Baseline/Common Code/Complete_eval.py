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
from pipeline_with_depth_treshold import *

IMAGES_TEST = '../Dataset/evaluation/images/'
LABELS_TEST = '../Dataset/evaluation/labels/'



def generate_predictions(yolo_model,depth_model,tti_classifier):
    y_pred = []
    y_true = []
    
    images = os.listdir(IMAGES_TEST)
    
    l = len(images)
    i = 1
    wrong_tti = 0
    wrong_class = 0
    no_pred = 0
    for img in images:
        print(f"Predicting image {i}/{l}")
        i+=1
        label = img.replace('.png', '.txt')
        # d = {(tool_class , tti_class) : is_tti}
        d = {}
        with open(LABELS_TEST + label, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                line = line.strip()
      
                tool_class = int(line[2])
                tti_class = int(line[4:])
                is_tti = int(line[0])
         
                d[(tool_class, tti_class)] = is_tti
                
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # _ , tti_predictions  = end_to_end_pipeline(IMAGES_TEST + img,yolo_model,depth_model,tti_classifier,device)
        _ , tti_predictions  = depth_treshold(IMAGES_TEST + img,yolo_model,depth_model)
        

        if len(tti_predictions) == 0 and len(d) != 0:
            y_pred.append(0)
            y_true.append(1)
            no_pred += 1
        
        for elem in tti_predictions:
            tool_class = elem['tool']['class']
            tissue = elem['tissue']['class']
            is_tti = elem['tti_class']
            
            found = False
            for pair in d:
                # print("pair: ", pair)
                # print((tool_class , tissue))
                if (tool_class , tissue) == pair:
                    found = True
                    if is_tti == d[pair]:
                        y_pred.append(1)
                        y_true.append(1)
                    else:
                        y_pred.append(0)
                        y_true.append(1)
                        wrong_tti += 1
            
            if not found:
                y_pred.append(0)
                y_true.append(1)
                wrong_class+=1
    
    print("Wrong tti: ",wrong_tti)
    print("Wrong classes: ",wrong_class)
    print("No preds: ",no_pred)
    return y_pred , y_true




if __name__ == "__main__":
    model = load_yolo_model('./runs_OLD_DATASET/segment/train/weights/best.pt')
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tti_class = ROIClassifier(2)
    # tti_class.load_state_dict(torch.load('ROImodel.pt',map_location=device))

    tti_class = ROIClassifierViT(2)
    tti_class.load_state_dict(torch.load('Vit2.pt',map_location=device))
    tti_class.to(device)
    
    y_pred , y_true = generate_predictions(model,depth,tti_class)

    # with open("data.pkl", "wb") as f:
    #     pickle.dump([y_true,y_pred], f)
        
    # with open("./data.pkl",'rb') as f:
    #     data = pickle.load(f)
    
    # y_true , y_pred = data
    zeros = 0
    ones = 0
    for i in y_true:
        if i == 0:
            zeros += 1
        else:
            ones += 1
            
            
    print("Ones: ", ones)
    print("Zeros: ", zeros)
    
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("f1 MACRO: ", f1_score(y_true, y_pred,average='macro'))
    # print("f1 WEIGHTED: ", f1_score(y_true, y_pred , average='weighted'))
    print("precision: ", precision_score(y_true, y_pred))
    print("recall: ", recall_score(y_true, y_pred))
    print("confusion matric: ", confusion_matrix(y_true, y_pred))
    # print("Balanced accuracy: ", balanced_accuracy_score(y_true, y_pred))




    
 