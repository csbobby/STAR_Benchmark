# -----------------------------------------------------
# STAR Visulization Tools
# temp_qa_visulization : used in QA_generation_demo to show generated question/answer/video
# show_generated_qa : show QAs in STAR
# -----------------------------------------------------

import os
import pickle
import json
import IPython
import cv2

import matplotlib.mlab as mlab  
import numpy as np

from IPython.display import Video, HTML
from ipywidgets import interact
from matplotlib import cm
from .vis_utils import *

font = cv2.FONT_HERSHEY_SIMPLEX
obj_vocab, rel_vocab = get_vocab('../annotations/STAR_classes')
act_cls = get_act_cls('../annotations/STAR_classes')

def Vis_Meta_Info(data):
    print('QID:', data['question_id'], ', VID: ', data['video_id']) 

def Vis_Question_Answer_Options(data):

    print('\tQ:', data['question'],'\n')
    print('\tAnswer:', data['answer'])
    for c in data['choices']:
        if c['choice'] != data['answer']:
            print('\tOption:', c['choice'])
    print('\n')

def Vis_Video(data,raw_video_dir,save_video_dir):

    start = round(data['start'], 2) # start time 
    end = round(data['end'], 2) # end time
    video_id = data['video_id']

    in_path = raw_video_dir + video_id + '.mp4'
    out_path = save_video_dir + data['question_id'] + '.mp4'
    print('\tVideo Seg: ', str(start) + 's', '-', str(end) + 's')
    os.system('ffmpeg -y -ss %s -to %s -i %s -codec copy %s' % (str(start), str(end), in_path, out_path))
    IPython.display.display( Video(out_path, embed=True, height=300, html_attributes="controls muted autoplay"))

def Vis_Keyframes(data,fps,max_show_num,raw_frame_dir):

    all_frame_dir = raw_frame_dir +'/' +data['video_id']+'.mp4'
    trimmed_frame_ids = trim_keyframes(data,fps,max_show_num)
    frame_plot(trimmed_frame_ids,all_frame_dir)

def Vis_Box(data,fps,max_show_num,raw_frame_dir):
    all_frame_dir = raw_frame_dir +'/' +data['video_id']+'.mp4'
    trimmed_frame_ids = trim_keyframes(data,fps,max_show_num)
    select = []
    for id in trimmed_frame_ids:
        bbox_list = data['situations'][id]['bbox']
        bbox_labels = data['situations'][id]['bbox_labels']
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(bbox_list) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        color_ctlr=0
        frame = cv2.imread(all_frame_dir+'/'+id+'.png')

        for i,bbox in enumerate(bbox_list):
            if bbox is not None:
                x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
                frame = cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)), colors[color_ctlr], 2)
                frame = cv2.putText(frame,obj_vocab[int(bbox_labels[i][1:])] , (int(x1),int(y1)), font, 1.2, (255, 255, 255), 2)
                color_ctlr+=1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        select.append(frame)

    ipyplot.plot_images(select,max_images=len(select),img_width=150)


def Vis_Pose(data,fps,max_show_num,raw_frame_dir,pose_dir):

    all_frame_dir = raw_frame_dir +'/' +data['video_id']+'.mp4'
    trimmed_frame_ids = trim_keyframes(data,fps,max_show_num)
    select = []

    for id in trimmed_frame_ids:
        pose_file = pose_dir + data['video_id'] + '/' + id + '.json'
        try:
            pose = json.load(open(pose_file))['people'][0]['pose_keypoints_2d']
            frame = cv2.imread(all_frame_dir+'/'+id+'.png')
            frame = vis_keypoints(frame,pose)
        except:
            frame = cv2.imread(all_frame_dir+'/'+id+'.png')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        select.append(frame)

    ipyplot.plot_images(select,max_images=len(select),img_width=150)


def Vis_SituationGraph(data,max_show_num):
    frame_ids = sorted(data['situations'].keys())
    trimmed_frame_ids = sample_frames(frame_ids,max_show_num)
    for f in trimmed_frame_ids:
        actions = data['situations'][f]['actions']
        act_arr = [act_cls[act] for act in actions]
        print('{} Frame ID:'.format(trimmed_frame_ids.index(f)),f)
        print('Subgraph:')
        print('\t Actions:')
        print('\t\t',' ,'.join(act_arr))

        rel_ids = data['situations'][f]['rel_labels']
        print('\t Relationships:')
        for j,rel in enumerate(data['situations'][f]['rel_pairs']):
            print('\t\t',obj_vocab[int(rel[0][1:])],' ---- ',rel_vocab[int(rel_ids[j][1:])],' ---- ',obj_vocab[int(rel[1][1:])])

        print('\n')

def qa_visulization(data,fps,
    raw_video_dir, save_video_dir, 
    raw_frame_dir, pose_dir, max_show_num,
    vis_meta=False, vis_q_a_o=False, vis_v=False, vis_kf=False, vis_sg=False, vis_pose=False, vis_box=False, **kwargs):

    i=0

    while i<len(data):

        if vis_meta:
            Vis_Meta_Info(data[i])

        if vis_q_a_o:
            print('='*20,'Question & Answer & Options','='*20,'\n')
            Vis_Question_Answer_Options(data[i])

        if vis_v:
            print('='*20,'Trimmed Video','='*20,'\n')
            Vis_Video(data[i],raw_video_dir,save_video_dir)

        if vis_kf:
            print('='*20,'Keyframes','='*20,'\n')
            Vis_Keyframes(data[i],fps,max_show_num,raw_frame_dir)

        if vis_pose:
            print('='*20,'Pose','='*20,'\n')
            Vis_Pose(data[i],fps,max_show_num,raw_frame_dir,pose_dir)

        if vis_box:
            print('='*20,'Bounding Boxes','='*20,'\n')
            Vis_Box(data[i],fps,max_show_num,raw_frame_dir)

        if vis_sg:
            print('='*20,'Situation Graphs','='*20,'\n')
            Vis_SituationGraph(data[i],max_show_num)

        i = i + 1

def show_generated_qa(QA,show_by_vid,show_by_qtype,**kwargs):

    if show_by_vid:
        QA = group_by_vid(QA)
        if show_by_qtype:
            QA = split_qtypes_in_vid(QA)
            @interact
            def show_qa_by_vid(vid = QA.keys()):
                qa_v = QA[vid]
                @interact
                def show_qa_by_qtype(qtype = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']):
                    qa = qa_v[qtype]
                    if len(qa)==0:
                        print('No {} question in {}'.format(qtype,vid))
                    else:
                        @interact
                        def show_qa_by_qid(qid = range(0, len(qa))): 
                            qa_visulization([qa[qid]], **kwargs)
        else:
            @interact
            def show_qa_by_vid(vid = QA.keys()):
                qa = QA[vid]
                @interact
                def show_qa_by_qid(qid = range(0, len(qa))): 
                    qa_visulization([qa[qid]], **kwargs)



    elif show_by_qtype:
        QA = group_by_qtypes(QA)
        
        @interact
        def show_qa_by_qtype(qtype = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']):
            qa = QA[qtype]
            @interact
            def show_qa_by_qid(qid = range(0, len(qa))): 
                qa_visulization([qa[qid]], **kwargs)

    else:

        @interact
        def show_qa_by_qid(qid = range(0, len(QA))): 
            qa_visulization([QA[qid]], **kwargs)

def temp_qa_visulization(data, raw_video_dir, save_video_dir):
    pre_video_ids = []
    i = 0
    
    while i < len(data):     
        #skip the same video
        if data[i]['video_id'] in pre_video_ids:
            continue
        else:
            pre_video_ids.append(data[i]['video_id'])

        Vis_Meta_Info(data[i])

        Vis_Question_Answer_Options(data[i])

        Vis_Video(data[i],raw_video_dir,save_video_dir)

        i = i + 1

def show_select_qa(QA,vid,qid_,**kwargs):

    QA = select_by_vid(QA,vid)
    QA = select_by_qid(QA,qid_)

    qids = [qa['question_id'] for qa in QA]

    @interact
    def show_qa(qid = qids): 
        qa_visulization([QA[qids.index(qid)]], **kwargs)
            
        
        






