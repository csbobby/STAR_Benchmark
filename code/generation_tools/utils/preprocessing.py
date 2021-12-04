import numpy as np
import pandas as pd
from .load_save import *

def get_rows_by_count(video_data, cleaned_voc, col_name = 'controled_action_words', cnt = 1):
    rows = None
    for word in cleaned_verb_voc.keys().values:
        if rows is None:
            rows = video_data[col_name].str.count(word)>= cnt
        rows = rows | (video_data[col_name].str.count(word)>= cnt )

    print('Get', sum(rows), 'rows')
    return rows
    
def filter_videos(data, conditions, drop):
    print("Before filter_videos(), len:", len(data))
    
    if 'actions' in data.columns:
        data = data[~data['actions'].isnull()]
        print('After', 'filter null actions', len(data))
        
    for condition in conditions:
        if condition == 'scene':
            removed_scenes = ['Stairs', 'Bathroom', 'Garage', 'Pantry', 'Other'] # Closet / Walk-in closet / Spear closet, Laundry room, Hallway, 
            data = data[~data[condition].isin(removed_scenes)] 
            print('After', condition, 'is not in removed_scenes: ', len(data))
            print('\t', removed_scenes)
        elif condition == 'verified':
            data = data[data[condition].str.contains("Yes")]
        elif condition == 'quality': 
            data = data[data[condition]>2]
            print('After', condition, '>2: ', len(data))
        elif condition == 'relevance':
            data = data[data[condition]>3]
            print('After', condition, '>3: ', len(data))
        elif condition == 'length':
            data = data[data[condition]>4]
#             data = data[data[condition]<13]
            print('After', condition, '>4s: ', len(data))
        elif condition == 'actions_num':
            data = data[data[condition]>1]
#             data = data[data[condition]<13]
            print('After', condition, '>1: ', len(data))
        elif condition == 'objects_num':
            data = data[data[condition]>1]
#             data = data[data[condition]<13]
            print('After', condition, '>1: ', len(data))
        
    
    print('Selected Scenes:')
    for s in set(data['scene']):
        print('\t', s)
    if drop:
        avoid_list = ['scene', 'actions_num', 'objects_num', 'length']
        for col in avoid_list:
            if col in conditions:
                conditions.remove(col)
        processed_data = data.drop(conditions, axis=1)
        print('Drop cols:', conditions)
    
    print("After filter_videos(), len:", len(processed_data))
    return processed_data


def filter_video_samples(data):
    
    data = data[data['actions'].str.len()>0]
#     data = data[data['actions'].str.len()>0]

#     processed_data = {}
#     for idx, each in data.iterrows():
#         action_list = each['actions'].split(";")
#         action_list = [x.split() for x in action_list]
#         processed_data[each['id']] = (action_list, each['length'], each['scene'])
#     return processed_data
    data = data[['id', 'actions', 'scene', 'subject'
                 , 'ordered_action_ids', 'action_words', 'verbs', 'objs', 'controled_action_words', 'descriptions'
                 , 'length']] #, 'action_segments']]
    return data # processed_data

def extract_action_dict(data,verb_map,obj_map,action_space):
    
    data = data[['id', 'actions', 'length', 'scene']]#, 'action_segments']]
    data = data[data['actions'].str.len()>0]
    verb_voc = list(set(action_space['verb']))
    obj_voc = list(set(action_space['object']))
    act_voc = list(set(action_space['action_id']))
    processed_data = {}
    for idx, each in data.iterrows():
        action_list = each['actions'].split(";")
        action_list = [x.split() for x in action_list]
        filtered_action_list = []
        for action in action_list:
            if (action[0] in act_voc) and (obj_map[action[0]] in obj_voc) and (verb_map[action[0]] in verb_voc):
                filtered_action_list.append(action)
        if filtered_action_list == []:
            continue
        else:
            processed_data[each['id']] = [filtered_action_list, each['length'], each['scene']]
    return processed_data


def denoise(action_list,verb_map,obj_map, max_time,min_time,max_action_id,eliminated_action_id,specific_verb,specific_obj):
    tmp_action_list = []
    for a in action_list:
        # remove actions with id larger than 145
        if (int(a[0][1:4]) > max_action_id):
            continue
        if (a[0] in eliminated_action_id):
            continue
        if  (verb_map[a[0]] in specific_verb):
            continue

        if  (obj_map[a[0]] in specific_obj):
            continue
        
        if ((float(a[2]) - float(a[1])) > max_time):
            continue

        if ((float(a[2]) - float(a[1])) < min_time):
            continue
        
        tmp_action_list.append(a)
        
    return tmp_action_list


def spatial_feasibitity(processed_video_data_fea,verb_map, obj_map,action_data, condition_action_data,cleaned_obj_voc):
    # Video Selection for Spatial Feasibility
    target_verb =    'open,close,hold,put,take,throw,grasp,eat,wash,pour,drink,sit,lie'.split(',')
    spatial_change_verb = 'walk,run'.split(',') 
    feasibility_target_verbs = ',' + '|,'.join(target_verb)
    print('feasibility_target_verbs:\t', feasibility_target_verbs)
    videos_with_target_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_target_verbs, regex=True)

    feasibility_spatial_change_verbs = ',|'.join(spatial_change_verb) + ','
    print('feasibility_spatial_change_verbs:\t', feasibility_spatial_change_verbs)
    # feasibility_condition_verbs = 'sit'
    videos_with_condition_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_spatial_change_verbs, regex=True)

    processed_fea_video_data = processed_video_data_fea[ videos_with_target_verbs & videos_with_condition_verbs ]
    processed_fea_video_data = processed_fea_video_data[processed_fea_video_data['verbs'].str.count(',')>1]

    print('Selected videos', len(processed_fea_video_data))

    # Use extract_action_dict to filter action_ids in QA generation
    processed_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map,action_data)
    processed_condition_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map, condition_action_data)
    return processed_action_dict_fea, processed_condition_action_dict_fea


def temporal_feasibitity(processed_video_data_fea,verb_map, obj_map,action_data, condition_action_data,cleaned_obj_voc):
    # Add Supportive and Conflict Rules
    target_verb =    'open,close,hold,put,take,throw,grasp,eat,wash,pour,drink,sit,lie'.split(',')
    v_move_temp = 'walk,verb|stand,verb|run,verb|sit,verb|lie,verb|'
    v_open = 'close,open|walk,open|stand,open|run,open|sit,open|lie,open|'
    v_close = v_open.replace('close,','open,').replace(',open',',close')
    v_move = ''
    for v in 'put,take,throw,drink'.split(','):
        v_move += 'open,put|' + v_move_temp.replace(',verb',',' + v)
    v_other = 'open,wash|' + 'take,throw|' + 'take,pour|' + 'take,drink'
    temporal_order_verb = v_open + v_close + v_move + v_other

    condition_verb = 'walk,run,sit,lie'.split(',')
    feasibility_condition_verbs = ',|'.join(condition_verb) + ','
    videos_with_condition_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_condition_verbs, regex=True)

    feasibility_target_verbs = ',' + '|,'.join(target_verb)
    print('feasibility_target_verbs:\t', feasibility_target_verbs)
    videos_with_target_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_target_verbs, regex=True)

    feasibility_temporal_order_verbs = temporal_order_verb # 'walk|stand|run'
    print('feasibility_temporal_order_verbs:\t', feasibility_temporal_order_verbs)
    videos_with_temporal_order_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_temporal_order_verbs, regex=True)

    processed_fea_video_data = processed_video_data_fea[ videos_with_target_verbs & videos_with_condition_verbs ]
    processed_fea_video_data = processed_fea_video_data[processed_fea_video_data['verbs'].str.count(',')>1]
    print('Selected videos', len(processed_fea_video_data))

    # Use extract_action_dict to filter action_ids in QA generation
    processed_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map,action_data)
    processed_condition_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map,condition_action_data)
    return processed_action_dict_fea, processed_condition_action_dict_fea

def compositional_obj_feasibility(processed_video_data_fea,verb_map, obj_map,action_data,condition_action_data,cleaned_obj_voc):

    target_verb =    'put,take,throw,grasp,eat,wash,pour,drink,lie'.split(',')
    duplicate_verb = [v+','+v for v in target_verb]

    feasibility_duplicate_verbs = '|'.join(duplicate_verb)
    print('feasibility_duplicate_verbs:\t', feasibility_duplicate_verbs)
    videos_with_condition_verbs = processed_video_data_fea['verbs'].str.contains(feasibility_duplicate_verbs, regex=True)

    processed_fea_video_data = processed_video_data_fea[ videos_with_condition_verbs ]
    processed_fea_video_data = processed_fea_video_data[processed_fea_video_data['verbs'].str.count(',')>1]
    print('Selected videos', len(processed_fea_video_data))
    processed_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map, action_data)
    processed_condition_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map, condition_action_data)
    return processed_action_dict_fea, processed_condition_action_dict_fea


def get_rows_by_count(video_data, cleaned_voc, col_name = 'controled_action_words', cnt = 1):
    rows = None
    for word in cleaned_voc.keys().values:
        if rows is None:
            rows = video_data[col_name].str.count(word)>= cnt
        rows = rows | (video_data[col_name].str.count(word)>= cnt )

    print('Get', sum(rows), 'rows')
    return rows

def compositional_verb_feasibility(processed_video_data_fea,verb_map, obj_map,action_data,condition_action_data,cleaned_obj_voc):
    rows_with_same_objs = get_rows_by_count(processed_video_data_fea, cleaned_obj_voc, 'controled_action_words', 2)
    processed_fea_video_data = processed_video_data_fea[rows_with_same_objs]
    print('Selected videos', len(processed_fea_video_data))
    processed_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map,action_data)
    processed_condition_action_dict_fea = extract_action_dict(processed_fea_video_data,verb_map, obj_map, condition_action_data)
    return processed_action_dict_fea, processed_condition_action_dict_fea


