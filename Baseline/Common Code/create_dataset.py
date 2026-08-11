import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset
import cv2
import json
from torchvision import transforms
from PIL import Image
import re
import random
import shutil
import torchvision.transforms as T


# Dataset organization:
# TODO
"""
/dataset/
  /images/
    /train/
      videoXXX_frame_000001.png
      videoXXX_frame_000002.png
      ...
    /val/
  /tti_labels/
    /train/
      videoXXX_frame_000001.txt
      videoXXX_frame_000002.txt
      ...
    /val/
  /instrument_labels/
    /train/
      videoXXX_frame_000001.txt
      videoXXX_frame_000002.txt
      ...
    /val/
"""
#1. Images extracted from videos and labelled by frame_id and arranged as above:
#1.1. Images are named by video and frame idx.
#1.2. They are organized into train/val/test

#2. labels for TTI extracted from the json to .txt file and organized as above:
#2.1. Each file represent a single frame.
#2.2. Each file contains one or multiple line labels for one or multiple TTI in that frame.
#2.3 The format for each label is:
#   <interaction> <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#2.4. The <interaction> label is a binary of 0 or 1 for if there is a TTI or not.

#3. labels for Instrument extracted from the json to .txt file and organized as above:
#3.1. Each file represent a single frame.
#3.2. Each file contains one or multiple line labels for one or multiple Instrument in that frame.
#3.3 The format for each label is:
#   <interaction> <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#3.4. The <interaction> label is a binary of 0 or 1 for if the instrument is involved in TTI or not.


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count


def _load_frame(cap, frame_idx, transform = None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #return transform(Image.fromarray(frame))
    return Image.fromarray(frame)


def to_tool_id(name):
    if name == None :
        return 0
    name_to_id = {
        'unknown_tool': 0,
        'dissector': 1,
        'scissors': 2,
        'suction': 3,
        'grasper 3': 4,
        'harmonic': 5,
        'grasper': 6,
        'bipolar': 7,
        'grasper 2': 8,
        'cautery (hook, spatula)': 9,
        'ligasure': 10,
        'stapler': 11,        
    }
    name = name.lower()
    if name not in name_to_id.keys():
      return 0
    return name_to_id[name]


def to_tti_id(name):
    if name == None:
      return 12
    name_to_id = {
        'unknown_tti': 12,
        'coagulation': 13,
        'other': 14,
        'retract and grab': 15,
        'blunt dissection': 16,
        'energy - sharp dissection': 17,
        'staple': 18,
        'retract and push': 19,
        'cut - sharp dissection': 20,
    }
    name = name.lower()
    if name not in name_to_id.keys():
        return 12
    
    return name_to_id[name]



"""CODE TO CREATE THE DATASET"""

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()


def create_dataset():
    file_path   = 'Dataset'
    videos_path = os.path.join(file_path, 'video_dataset/videos/val')
    json_folder = os.path.join(file_path, 'video_dataset/labels/val')


    i = 1
    videos = os.listdir(videos_path)

    for video in videos:
        print(f"Processing video {i}/{len(videos)}: {video}")

        cap, frame_count = _load_video(os.path.join(videos_path, video))


        base_video = os.path.splitext(video)[0]
        key_video  = normalize(base_video)

        matched_json = None
        for fname in os.listdir(json_folder):
            name_no_ext = os.path.splitext(fname)[0]
            if normalize(name_no_ext) == key_video:
                matched_json = os.path.join(json_folder, fname)
                break

        if not matched_json:
            print(f"[WARNING] No JSON file for «{video}». Skipped")
            i += 1
            continue

        
        with open(matched_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_indices = list(data['labels'].keys())
            frame_indices = [int(idx) for idx in frame_indices if len(data['labels'][idx]) != 0]

        for idx in frame_indices:
            if int(idx) > 150:
                continue
            frame = _load_frame(cap, idx)
            out_img = os.path.join(file_path, 'yolo_dataset', 'images', 'val',f'video{i:04d}_frame{idx:04d}.png')
            frame.save(out_img)
            print(f"Saved {out_img}") 


            to_write = ''
            # print(len(data['labels'][str(idx)]))
            
            for j in range(len(data['labels'][str(idx)])):
                current_dict = data['labels'][str(idx)][j]
                
                
                dict_len = len(current_dict.keys())
                if dict_len == 4:
                    
                    is_tti = data['labels'][str(idx)][j]['is_tti']
                    interaction_type = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
                    to_write += str(is_tti) + ' ' + str(interaction_type) 
                    tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                    
                    for vertex in tti_polygon.keys():
                        x = data['labels'][str(idx)][j]['tti_polygon'][vertex]['x']
                        y = data['labels'][str(idx)][j]['tti_polygon'][vertex]['y']
                        to_write += ' ' + str(x) + ' ' + str(y)
                        
                    to_write += '\n'
                    
                elif dict_len == 3:
                    
                    is_tti = data['labels'][str(idx)][j]['is_tti']
                    non_interaction_tool = to_tool_id(data['labels'][str(idx)][j]['non_interaction_tool'])
                    to_write += str(is_tti) + ' ' + str(non_interaction_tool)
                    instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                    
                    for vertex in instrument_polygon.keys():
                        x = instrument_polygon[vertex]['x']
                        y = instrument_polygon[vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                        
                    to_write += '\n'
                    
                elif dict_len == 2:
                    instr_type = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])
                    
                    to_write += str(0) + ' ' + str(instr_type) 
                    instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                    
                    for vertex in instrument_polygon.keys():
                        x = instrument_polygon[vertex]['x']
                        y = instrument_polygon[vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                        
                    to_write += '\n'
                    
            with open(file_path+'/yolo_dataset/labels/val/'+ f'video{i:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
                f.write(to_write)   

        i += 1


def move_file(fname, split):
    base_dir = './Dataset/video_dataset'
    
    # image
    src_img = os.path.join(base_dir, 'videos', 'train', fname)
    dst_img = os.path.join(base_dir, 'videos', split, fname)
    shutil.move(src_img, dst_img)

    # labels
    lbl = fname.replace('.mp4', '.json')
    src_lbl = os.path.join(base_dir, 'labels', 'train', lbl)
    dst_lbl = os.path.join(base_dir, 'labels', split, lbl)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)


def split_dataset():

    base_dir    = './Dataset/video_dataset'
    folders     = ['videos', 'labels']
    splits      = ['train', 'val', 'test']
    train_split = 0.75
    val_split   = 0.2
    # test_split (0.1)


    for folder in folders:
        for split in splits:
            path = os.path.join(base_dir, folder, split)
            os.makedirs(path, exist_ok=True)


    img_train_dir = os.path.join(base_dir, 'videos', 'train')
    all_imgs = [f for f in os.listdir(img_train_dir) if f.lower().endswith('.mp4')]

    # shuffle per random
    random.seed(42)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    print("The whole dataset is composed by " + str(n) + " videos")
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test = n - n_train - n_val

    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train:n_train + n_val]
    test_imgs  = all_imgs[n_train + n_val:]


    for fname in train_imgs:
        move_file(fname, 'train')
    for fname in val_imgs:
        move_file(fname, 'val')
    for fname in test_imgs:
        move_file(fname, 'test')

    print(f"Final distribution: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    
    
if __name__ == "__main__":
    create_dataset()
    # split_dataset()
    
    PATH_VIDEO = './Dataset/video_dataset/videos/train/'
    path_labels = './Dataset/video_dataset/labels/train/'
    # shutil.move(path_labels + 'Adnanset-Lc 122-003.json', './Dataset/video_dataset/labels/test/')

    
    # labels = os.listdir(path_labels)
    # i=0
    # c = 0

    # to_remove = []

    # for label in labels:
  
    #     print(f"processing label {c}/{len(labels)}")
    #     c+=1
    #     key_label  = os.path.splitext(label)[0]
    #     key_label = normalize(key_label)
    #     matched_video = None
    #     found = False

    #     for video in os.listdir(PATH_VIDEO):
    #         name_no_ext = os.path.splitext(video)[0]
    #         if normalize(name_no_ext) == key_label:
    #             matched_video = os.path.join(PATH_VIDEO, name_no_ext)
    #             found = True
    #             break
        
    #     if not found:
    #         to_remove.append(label)
    #         i+=1
    
    # for elem in to_remove:
    #     os.remove(path_labels+elem)
    # print("Eliminated " + str(len(to_remove)) + " labels")
          

    
