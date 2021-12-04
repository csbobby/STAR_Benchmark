import numpy as np
import json
np.random.seed(126)

def split_by_qtype(QA,return_dict=False):
    qa_dict = {'Interaction':[],'Sequence':[],'Prediction':[],'Feasibility':[]}
    for qa in QA:
        qa_dict[qa['question_id'].split('_')[0]].append(qa)
    if return_dict:
        return qa_dict
    return qa_dict['Interaction'], qa_dict['Sequence'], qa_dict['Prediction'], qa_dict['Feasibility']

def split_by_vid(QA,return_dict=False):
    qa_dict = {}
    for qa in QA:
        if qa['video_id'] not in qa_dict:
            qa_dict[qa['video_id']] = [qa]
        else:
            qa_dict[qa['video_id']].append(qa)
    return qa_dict

def split_by_temp(QA,return_dict=False):
    qa_dict = {}
    for qa in QA:
        try:
            qa_dict[qa['question_id'].split('_')[1]].append(qa)
        except:
            qa_dict[qa['question_id'].split('_')[1]] = [qa]
    return qa_dict

def merge_qa(QA,qa,total_cnt,type_cnt):
    return QA+qa, total_cnt+len(qa), type_cnt+len(qa)

def split_dataset(QA,control_ratio,reverse=False,train_split=0.6, val_split=0.2, test_split=0.2):
    total_num = len(QA)*control_ratio
    train_num, test_num = int(total_num*train_split), int(total_num*test_split)
    val_num = total_num-train_num-test_num
    int_QA, seq_QA, pre_QA, fea_QA = split_by_qtype(QA)
    total_int, total_seq, total_pre, total_fea = len(int_QA), len(seq_QA), len(pre_QA), len(fea_QA)
    train_int,test_int = int(total_int*train_split),int(total_int*test_split)
    train_seq,test_seq = int(total_seq*train_split), int(total_seq*test_split)
    train_pre,test_pre = int(total_pre*train_split), int(total_pre*test_split)
    train_fea, test_fea = int(total_fea*train_split), int(total_fea*test_split)
    val_int, val_seq, val_pre, val_fea = total_int-train_int-test_int, total_seq-train_seq-test_seq, total_pre-train_pre-test_pre, total_fea-train_fea-test_fea
    
    vid_dict = split_by_vid(QA)
    vid_qa_num_dict = {}
    for key in vid_dict:
        vid_qa_num_dict[key] = len(vid_dict[key])
    sorted_vid = [v[0] for v in sorted(vid_qa_num_dict.items(),key=lambda x:(x[1],x[0]),reverse=reverse)]
    sorted_vid = list(np.random.permutation(sorted_vid))

    qa_train,qa_val,qa_test = [],[],[]
    val_test_vids = []
    int_cnt,seq_cnt,pre_cnt,fea_cnt,train_cnt = 0,0,0,0,0
    
    for video in sorted_vid:
        int_qa, seq_qa, pre_qa, fea_qa = split_by_qtype(vid_dict[video])
        if train_cnt<train_num:
            if (int_cnt<train_int):
                qa_train, train_cnt, int_cnt = merge_qa(qa_train,int_qa,train_cnt,int_cnt)
                if train_cnt>train_num:
                    continue
            if (seq_cnt<train_seq):
                qa_train, train_cnt, seq_cnt = merge_qa(qa_train,seq_qa,train_cnt,seq_cnt)
                if train_cnt>train_num:
                    continue
            if (pre_cnt<train_pre):
                qa_train, train_cnt, pre_cnt = merge_qa(qa_train,pre_qa,train_cnt,pre_cnt)
                if train_cnt>train_num:
                    continue
            if (fea_cnt<train_fea):
                qa_train, train_cnt, fea_cnt = merge_qa(qa_train,fea_qa,train_cnt,fea_cnt)
                if train_cnt>train_num:
                    continue
        else:
            val_test_vids.append(video)
   
    int_cnt,seq_cnt,pre_cnt,fea_cnt,val_cnt = 0,0,0,0,0
    test_vids = []
    for video in val_test_vids:
        int_qa, seq_qa, pre_qa, fea_qa = split_by_qtype(vid_dict[video])
        if val_cnt<val_num:
            if (int_cnt<val_int):
                qa_val, val_cnt, int_cnt = merge_qa(qa_val,int_qa,val_cnt,int_cnt)
                if val_cnt>val_num:
                    continue
            if (seq_cnt<val_seq):
                qa_val, val_cnt, seq_cnt = merge_qa(qa_val,seq_qa,val_cnt,seq_cnt)
                if val_cnt>val_num:
                    continue
            if (pre_cnt<val_pre):
                qa_val, val_cnt, pre_cnt = merge_qa(qa_val,pre_qa,val_cnt,pre_cnt)
                if val_cnt>val_num:
                    continue
            if (fea_cnt<val_fea):
                qa_val, val_cnt, fea_cnt = merge_qa(qa_val,fea_qa,val_cnt,fea_cnt)
                if val_cnt>val_num:
                    continue
        else:
            test_vids.append(video)
            
    int_cnt,seq_cnt,pre_cnt,fea_cnt,test_cnt = 0,0,0,0,0
    for video in test_vids:
        int_qa, seq_qa, pre_qa, fea_qa = split_by_qtype(vid_dict[video])
        if (int_cnt<test_int):
            qa_test, test_cnt, int_cnt = merge_qa(qa_test,int_qa,test_cnt,int_cnt)
        if (seq_cnt<test_seq):
            qa_test, test_cnt, seq_cnt = merge_qa(qa_test,seq_qa,test_cnt,seq_cnt)
        if (pre_cnt<test_pre):
            qa_test, test_cnt, pre_cnt = merge_qa(qa_test,pre_qa,test_cnt,pre_cnt)
        if (fea_cnt<test_fea):
            qa_test, test_cnt, fea_cnt = merge_qa(qa_test,fea_qa,test_cnt,fea_cnt) 

    print('Train:',len(qa_train))
    print('Val:',len(qa_val))
    print('Test:',len(qa_test))
            
    return qa_train,qa_val, qa_test
