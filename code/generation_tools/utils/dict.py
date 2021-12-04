import csv
import numpy as np
import pandas as pd


def objName2IdDict(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_objectclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            names = mapping[1].split('/')
            for name in names:
                map[name] = mapping[0]
    return map

def verbName2IdDict(csv_dir):
    map = {}
    with open(csv_dir + "/Charades_v1_verbclasses.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            names = mapping[1].split('/')
            for name in names:
                map[name] = mapping[0]
    return map


def inverseActionDict(csv_dir):
    # key: oxxx vxxx  value: cxxx
    map = {}
    with open(csv_dir + "/Charades_v1_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            k = mapping[1] + ' ' + mapping[2]
            v = mapping[0]
            map[k] = v
    return map


def frame2time(frame, video_id, fps_dict, video_length):
    return int(frame) / fps_dict[video_id + '.mp4']


def intToActionId(str_number):
    str_number = str(str_number)
    while len(str_number) < 3:
        str_number = '0' + str_number
    return 'c' + str_number
