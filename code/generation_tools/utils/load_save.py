import os
import pickle
import cv2
import csv
import numpy as np
import pandas as pd
import json
from sklearn.utils import shuffle


def load_annotations(annotation_dir):
    with open(os.path.join(annotation_dir, 'object_bbox_and_relationship.pkl'), 'rb') as f:
        object_anno = pickle.load(f)

    return object_anno


# split annotation into {Video_No:{Frame_No}} from
def anno_data_split(object_anno):
    anno_video_split = {}
    for k in object_anno.keys():
        s = k.split('/')
        video_id = s[0][:5]
        frame_id = s[1][:6]
        if not anno_video_split.__contains__(video_id):
            anno_video_split[video_id] = {frame_id: object_anno[k]}
        else:
            anno_video_split[video_id][frame_id] = object_anno[k]
    return anno_video_split

def read_video(file_path):
    video_cap = cv2.VideoCapture(file_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        video_FPS = video_cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        video_FPS = video_cap.get(cv2.CAP_PROP_FPS)
    return video_FPS


def load_csv(csv_dir):
    data = None
    for filename in ["/Charades_v1_test.csv", "/Charades_v1_train.csv"]:
        if data is None:
            data = pd.read_csv(csv_dir + filename)
        else:
            tmp_data = pd.read_csv(csv_dir + filename)
            data = pd.concat([data, tmp_data])
            data = data.reset_index(drop=True)
        print('After load_csv(), len:', len(data))
#         with open(csv_dir + filename) as f:
#             f_csv = csv.reader(f)
#             for (i, row) in enumerate(f_csv):
#                 if i == 0:
#                     continue
#                 data.append(row)
    return data


def load_action_mapping(csv_dir):

    dict = {}
    with open(csv_dir + "/action_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            tag = line[0:4]
            description = line[5:-1]
            dict[tag] = description
    return dict


def load_obj_mapping(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[1]
    obj_map = {}
    with open(csv_dir + "/Charades_v1_objectclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            obj_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = obj_map[map[k]]
    return map

def load_verb_mapping(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[2]
    verb_map = {}
    with open(csv_dir + "/Charades_v1_verbclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            verb_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = verb_map[map[k]]
    return map


def load_all_json_in_folder(path):
    qa = []
    json_files = os.listdir(path)
    for file in json_files:
        if 'json' in file:
            qa.extend(json.load(open(path+file)))
    return qa

def load_json(filename):
    return json.load(open(filename))

def write_json(QAList, path):
    with open(path, 'w') as f:
        json.dump(QAList, f)
    f.close()

# Saving
def save_qa_data(QAList, qtype, template,save_data_dir,
    save_qa_json = True,
    save_shuffle = True):
    if QAList is None or len(QAList) == 0:
        print('QAList is empty, nothing saved.')
        return
    if save_shuffle:
        np.random.seed(RANDOM_SEED) # fix seed
        QAList = list(np.random.permutation(QAList)) # remembered shuffle for saving
    if save_qa_json:
        json_save_path = os.path.join( save_data_dir, qtype , '{}_T{}.json'.format(qtype, str(template)) )
        write_json(QAList, json_save_path)
        print('Saved QA Data to: ', json_save_path)
