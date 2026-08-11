import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import numpy as np
from transformers import pipeline
from PIL import Image
from Model import ROIClassifier
import matplotlib.pyplot as plt
from pipeline import *
from training_NN_pipe import _load_video, _load_frame, normalize , get_polygon ,  parse_mask_string
import os
from create_dataset import to_tool_id , to_tti_id
import json
from sklearn.metrics import accuracy_score , f1_score ,precision_recall_curve ,precision_score ,recall_score ,confusion_matrix ,balanced_accuracy_score
import pickle
from VIT import ROIClassifierViT

LABELS_PATH = '../Dataset/video_dataset/labels/val/'
VIDEOS_PATH = '../Dataset/video_dataset/videos/val/'

IMAGES_TEST = '../Dataset/evaluation/images/'
LABELS_TEST = '../Dataset/evaluation/labels/'


def create_test():
    
    videos = os.listdir(VIDEOS_PATH)

    file_path   = 'Dataset'
    count = 0
    
    
    for video in videos:
       
        print(f"Processing video {count}/{len(videos)}: {video}")
        cap, frame_count = _load_video(os.path.join(VIDEOS_PATH, video))

        base_video = os.path.splitext(video)[0]
        key_video  = normalize(base_video)


        matched_json = None
        for fname in os.listdir(LABELS_PATH):
            name_no_ext = os.path.splitext(fname)[0]
            if normalize(name_no_ext) == key_video:
                matched_json = os.path.join(LABELS_PATH, fname)
                break

        if not matched_json:
            print(f"[WARNING] No JSON file for «{video}». Skipped")
            count += 1
            continue

        with open(matched_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_indices = list(data['labels'].keys())
            frame_indices = [int(idx) for idx in frame_indices if len(data['labels'][idx]) != 0]


        for idx in frame_indices:
            to_write = ''
            if int(idx) > 150:
                continue
            frame = _load_frame(cap, idx)
            
            W, H = frame.size
            
            len_dict = len(data['labels'][str(idx)])
            
            if len_dict < 2:
                continue

            frame.save(file_path+'/evaluation/images/'+ f'video{count:04d}_frame{idx:04d}.png', format='PNG')
            
            tissue_mask = None
            tool_mask = None
            non_tool_mask = None
            tool_class = 0
            tti_class = 0
            non_tool_class = 0
            
            if len_dict == 2:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                            if data['labels'][str(idx)][j]['is_tti'] == 1:
                                tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                                tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
       
                                tti_polygon_str = get_polygon(tti_polygon)
                                tissue_mask = parse_mask_string(tti_polygon_str,H,W)                   
                                
                            else:
                                continue
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                        tool_class = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])
                       
                        

                if tool_mask is not None and tissue_mask is not None:
                    to_write += '1' + ' ' + str(tool_class) + ' ' + str(tti_class)  + ' ' + '\n'
                    # y_train.append(1)
          
          
            if len_dict == 3:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type']) 
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                        else:
                            non_int_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            non_int_polygon_str = get_polygon(non_int_polygon)
                            non_tool_mask = parse_mask_string(non_int_polygon_str,H,W)
                            non_tool_class = to_tool_id(data['labels'][str(idx)][j]['non_interaction_tool'])
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                        tool_class = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])

                if tool_mask is not None and tissue_mask is not None:
                    to_write += '1' + ' ' + str(tool_class) + ' ' + str(tti_class)  + ' ' + '\n'
                    # y_train.append(1)   
                if non_tool_mask is not None and tissue_mask is not None:       
                    to_write += '0' + ' ' + str(non_tool_class) + ' ' + str(tti_class)  + ' ' + '\n'
                    # y_train.append(0)          
                
            
            if len_dict == 4:
                pair_list = []
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                            interaction_tool_name = data['labels'][str(idx)][j]['interaction_tool']
                            pair_list.append((tissue_mask, interaction_tool_name , tti_class))
        
                for pair in pair_list:
                    for j in range(len_dict):
                        if not bool(data['labels'][str(idx)][j].keys()):
                            continue
                        if 'is_tti' not in data['labels'][str(idx)][j].keys():
                            tool_name = pair[1]
                            tissue_mask = pair[0]
                            tti_class = pair[2]
                            instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            instrument_polygon_str = get_polygon(instrument_polygon)
                            tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                            tool_class = to_tool_id(tool_name)
                            
                            if tissue_mask is not None and tool_mask is not None:
                                if tool_name == data['labels'][str(idx)][j]['instrument_type']:
                                    # y_train.append(1)   
                                    to_write += '1' + ' ' + str(tool_class) + ' ' + str(tti_class)  + ' ' + '\n'
                                else:
                                    # y_train.append(0) 
                                    to_write += '0' + ' ' + str(to_tool_id(data['labels'][str(idx)][j]['instrument_type'])) + ' ' + str(tti_class)  + ' ' + '\n'
              
            with open(file_path+'/evaluation/labels/'+ f'video{count:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
                f.write(to_write) 
                                                        
        count += 1
        
           

def generate_predictions(yolo_model,depth_model,tti_classifier):
    y_pred = []
    y_true = []
    
    images = os.listdir(IMAGES_TEST)
    
    l = len(images)
    i = 1
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
        _ , tti_predictions  = end_to_end_pipeline(IMAGES_TEST + img,yolo_model,depth_model,tti_classifier,device)
        for elem in tti_predictions:
            tool_class = elem['tool']['class']
            tissue = elem['tissue']['class']
            is_tti = elem['tti_class']
            
            for pair in d:
                # print("pair: ", pair)
                # print((tool_class , tissue))
                if (tool_class , tissue) == pair:
                    y_pred.append(is_tti)
                    y_true.append(d[pair])

            
    return y_pred , y_true




if __name__ == "__main__":
    model = load_yolo_model('./runs_OLD_DATASET/segment/train/weights/best.pt')
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tti_class = ROIClassifier(2)
    # tti_class.load_state_dict(torch.load('ROImodel.pt',map_location=device))
    tti_class = ROIClassifierViT(2)
    tti_class.load_state_dict(torch.load('ViT2.pt',map_location=device))
   
    tti_class.to(device)
    
    y_pred , y_true = generate_predictions(model,depth,tti_class)

    # with open("data.pkl", "wb") as f:
    #     pickle.dump([y_true,y_pred], f)
        
    # with open("./data.pkl",'rb') as f:
    #     data = pickle.load(f)
    
    # y_true , y_pred = data
    

    # l = len(y_true)
    # c = 0
    # for i in range(l):
    #     if c == 930:
    #         break
    #     if y_true[i] == 1:
    #         del y_true[i]
    #         del y_pred[i]
    #     c+=1
        
        
    zeros = 0
    ones = 0
    for i in y_true:
        if i == 0:
            zeros += 1
        else:
            ones += 1
    
    print("number of ones: ",ones)
    print("number of zeros: ",zeros)
    
    
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("f1 MACRO: ", f1_score(y_true, y_pred,average='macro'))
    # print("f1 WEIGHTED: ", f1_score(y_true, y_pred , average='weighted'))
    print("precision: ", precision_score(y_true, y_pred))
    print("recall: ", recall_score(y_true, y_pred))
    print("confusion matric: ", confusion_matrix(y_true, y_pred))
    # print("Balanced accuracy: ", balanced_accuracy_score(y_true, y_pred))

    # create_test()

    
 