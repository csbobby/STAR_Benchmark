import json
import numpy as np
import cv2
import ipyplot
import matplotlib.pyplot as plt

def sample_frames(frame_ids, max_show_num):
    # sample frames from given frame IDs averagely according to max_show_num
    if max_show_num==0:
        return frame_ids
    max_show_num = min(len(frame_ids), max_show_num)
    interval = int(len(frame_ids)/max_show_num)
    return frame_ids[::interval]

def trim_keyframes(data,fps,max_show_num=4):
    frame_ids = list(sorted(data['situations'].keys()))
    trimmed_frame_ids = [frame for frame in frame_ids if int(frame)>=(data['start'])*fps[data['video_id']+'.mp4']+1 and int(frame)<(data['end'])*fps[data['video_id']+'.mp4']+1]
    trimmed_frame_ids = sample_frames(trimmed_frame_ids,max_show_num)
    return trimmed_frame_ids

def frame_plot(frame_list,frame_dir):
    select = []
    for i in range(len(frame_list)):
        frame = cv2.imread(frame_dir+'/'+frame_list[i]+'.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        select.append(frame)
    ipyplot.plot_images(select,max_images=len(select),img_width=150)

def vis_keypoints(img,kpts):

    link2 = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]
    x_ = kpts[0::3]
    y_ = kpts[1::3]
    v_ = kpts[2::3]
    x_max = np.float32(max(x_))
    x_min = np.float32(min(x_))
    y_max = np.float32(max(y_))
    y_min = np.float32(min(y_))
    x_len = x_max-x_min
    y_len = y_max-y_min
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(link2) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(link2)):
        order1, order2 = link2[i][0], link2[i][1]
        x1 =int(np.float32(x_[order1]))
        y1 =int(np.float32(y_[order1]))
        x2 =int(np.float32(x_[order2]))
        y2 =int(np.float32(y_[order2]))

        if v_[order1] > 0 and v_[order2] > 0:
            cv2.line(img,(x1,y1),(x2,y2),thickness=4,color=colors[i])
    i = 0
    for x, y, v in zip(x_, y_, v_):
        x = int(np.float32(x))
        y = int(np.float32(y))
        if v > 0:
            img = cv2.circle(
                img, (x, y),1, (0,0,255),-1)
            i = i+1

    return img

def group_by_vid(QA):
    qa_by_vid = {}
    vid = {}
    for qa in QA:
        if qa['video_id'] not in qa_by_vid:
            qa_by_vid[qa['video_id']] = [qa]
        else:
            qa_by_vid[qa['video_id']].append(qa)
    return qa_by_vid

def split_qtypes_in_vid(QA):

    for vid in QA:
        QA[vid] = group_by_qtypes(QA[vid])

    return QA

def group_by_qtypes(QA):

    qa_by_qtype = {
        "Interaction":[],
        "Sequence":[],
        "Prediction":[],
        "Feasibility":[]
    }
    for qa in QA:
        qa_by_qtype[qa['question_id'].split('_')[0]].append(qa)
            
    return qa_by_qtype

def select_by_vid(QA,vid):
    qa_select = []
    if vid =='':
        return QA

    for qa in QA:
        if qa['video_id']==vid:
            qa_select.append(qa)

    if qa_select == []:
        return QA

    return qa_select

def select_by_qid(QA,qid):

    selected_qa = []

    for qa in QA:
        if qa['question_id'] in qid:
            selected_qa.append(qa)

    if len(selected_qa)==0:
        return QA

    return selected_qa

def get_vocab(label_dir):
    obj_to_ind, ind_to_obj, ind_to_rel, rel_to_ind = {}, {}, {}, {}
    obj_vocab, rel_vocab =[], []

    with open(label_dir + "/object_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            obj_vocab.append(mapping.split(' ')[1])

    with open(label_dir + "/relationship_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            rel_vocab.append(mapping.split(' ')[1])


    return obj_vocab, rel_vocab

def get_act_cls(label_dir):

    dict = {}
    with open(label_dir + "/action_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            tag = line[0:4]
            description = line[5:-1]
            dict[tag] = description
    return dict

