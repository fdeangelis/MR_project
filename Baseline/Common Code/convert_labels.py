import json
import os

LABEL_PATH = "./Dataset/labels/"
OUTPUT_PATH = "./Dataset/out/"
# log = open("log.txt", "w", encoding='utf-8')

def convert_one_label(label : json, all_interaction_types, all_interaction_tools):
    j = json.load(open(LABEL_PATH + label))
    file_name = j[0]['data_title']
    data_hash = j[0]['data_hash']
    frames = j[0]['data_units'][data_hash]['labels'].keys()
    # print(frames)
    to_write = {"labels": {}}
    
    for k in frames:
        list_to_add = []
        
        polygon_number = len(j[0]['data_units'][data_hash]['labels'][k]['objects'])
        for i in range(polygon_number):
            key_dict = dict()
            
            polygon_name = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['name']
        
            if polygon_name == 'TTI Free Drawing':
                key_dict['is_tti'] = 1
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                interaction_type = None
                interaction_tool = None
                # print(j[0]['object_answers'][object_hash]['classifications'])
                
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    for l in range(len(j[0]['object_answers'][object_hash]['classifications'])):
                        if j[0]['object_answers'][object_hash]['classifications'][l]['name'] == "Instrument type":
                            interaction_tool = j[0]['object_answers'][object_hash]['classifications'][l]['answers'][0]['name']                         
                        elif j[0]['object_answers'][object_hash]['classifications'][l]['name'] == "Interaction type":
                            interaction_type = j[0]['object_answers'][object_hash]['classifications'][l]['answers'][0]['name']
                            
                key_dict['interaction_type'] = interaction_type
                # print(interaction_type, " --- ", label)
                # log.write(str(interaction_type) + " --- " + label + "\n")
                key_dict['interaction_tool'] = interaction_tool

                if interaction_type is not None:
                    all_interaction_types.add(interaction_type)
                if interaction_tool is not None:
                    all_interaction_tools.add(interaction_tool)


                # print(j[0]['data_units'][data_hash]['labels'][k]['objects'][i])
                key_dict['tti_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
            
            elif polygon_name == "No Interaction":
                key_dict['is_tti'] = 0
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                non_interaction_tool = None
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    non_interaction_tool = j[0]['object_answers'][object_hash]['classifications'][0]['answers'][0]['name']
                key_dict['non_interaction_tool'] = non_interaction_tool
                key_dict['instrument_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
            
            elif polygon_name == 'Instrument Shaft Lavel':
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                instrument_type = None
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    for l in range(len(j[0]['object_answers'][object_hash]['classifications'])):
                        if j[0]['object_answers'][object_hash]['classifications'][l]['name'] == "Instrument type":
                            instrument_type = j[0]['object_answers'][object_hash]['classifications'][l]['answers'][0]['name']
                key_dict['instrument_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
                key_dict['instrument_type'] = instrument_type
                
            
            
            
            list_to_add.append(key_dict)
        
            
            
        to_write['labels'][k] = list_to_add
    
    with open(OUTPUT_PATH+file_name[:-4]+'.json','w') as file:
        json.dump(to_write,file,indent=2)

    with open('interaction_types.txt', 'w', encoding='utf-8') as f_types:
        for t in all_interaction_types:
            f_types.write(f"{t}\n")

    with open('interaction_tools.txt', 'w', encoding='utf-8') as f_tools:
        for tool in all_interaction_tools:
            f_tools.write(f"{tool}\n")
    


if __name__ == "__main__":
    labels = os.listdir(LABEL_PATH)
    total_labels = len(labels)
    i = 1
    all_interaction_types = set()
    all_interaction_tools = set()
    for l in labels:
        print(f"Processing: {i}/{total_labels}")
        # print(l)
        convert_one_label(l, all_interaction_types, all_interaction_tools)
        i+=1
        
    # log.close()


