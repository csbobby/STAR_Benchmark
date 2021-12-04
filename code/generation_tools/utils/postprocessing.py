import numpy as np
import json
import os
import copy
import math

RANDOM_SEED = 626
np.random.seed(RANDOM_SEED)

def demo_postprocess(QAList, selected_video_ids = [], show_shuffle = True):
    if QAList is None or len(QAList) == 0:
        print('QA List is null.')
        return None 
    # select
    if selected_video_ids is not None and len(selected_video_ids) > 0:
        QAList = [qa for qa in QAList if qa['video_id'] in selected_video_ids ]
    # shuffle
    if show_shuffle:
        QAList = list(np.random.permutation(QAList)) # default shuffle for showing
    
    return QAList

def QA_postprocess(QA,thresh,ratio,dtype,qtype,debias_type='action'):
    #print('{} {} QA num before debiasing: '.format(qtype, dtype),len(QA)) 
    if dtype == 'train':
        qa_breaked = QA
        #print('{} {} QA num after debiasing: '.format(qtype, dtype),len(qa_breaked))
    else:
        qa_breaked = debiasing(QA,thresh,ratio, qtype, dtype, debias_type)
        qa_breaked = debiasing(qa_breaked,0.2, 1.0, qtype, dtype, 'answer')
        #print('{} {} QA num after debiasing: '.format(qtype, dtype),len(qa_breaked))
        #qa_breaked = breaking_shortcuts(QA,mode)
    #print('QA num after breaking shortcuts: ',len(qa_breaked))
    #print('{} {} Down Sampling Ratio: '.format(qtype, dtype), (len(QA)-len(qa_breaked))/len(QA))
    return qa_breaked

def get_answer_space(QA,save_dir):
    train_ans_space = {}
    for qa in QA:
        qid = qa['question_id']
        qtype, qtemp = qid.split('_')[0], qid.split('_')[1] 
        answer = qa['answer']
        if qtype not in train_ans_space:
            train_ans_space[qtype] = {}
        if qtemp not in train_ans_space[qtype]:
            train_ans_space[qtype][qtemp] = []

        if answer not in train_ans_space[qtype][qtemp]:
            train_ans_space[qtype][qtemp].append(answer)
    with open(save_dir,'w') as f:
        f.write(json.dumps(train_ans_space))
    return train_ans_space

def get_answer_frequency(QA,save_dir):
    train_fre_space = {}
    for qa in QA:
        qid = qa['question_id']
        qtype, qtemp = qid.split('_')[0], qid.split('_')[1] 
        q_key = qa['question_keyword']
        a_key = qa['answer_keyword']
        if qtype not in train_fre_space:
            train_fre_space[qtype] = {}
        if qtemp not in train_fre_space[qtype]:
            train_fre_space[qtype][qtemp] = {}
        if q_key not in train_fre_space[qtype][qtemp]:
            train_fre_space[qtype][qtemp][q_key]={}
        if a_key not in train_fre_space[qtype][qtemp][q_key]:
            train_fre_space[qtype][qtemp][q_key][a_key] = 1
        else:
            train_fre_space[qtype][qtemp][q_key][a_key] += 1
    with open(save_dir,'w') as f:
        f.write(json.dumps(train_fre_space))
    return train_fre_space

def static_distribution(QA):
    # QA [list] : generated qa pairs  
    Q2A_stat = {}
    answer_stat = {}
    action_stat = {}
    
    for qa in QA:
        template = qa['question_id'].split('_')[1] 
        a_keyword = qa['answer_keyword']
        q_keyword = qa['question_keyword']
        answer = qa['answer']
        act_id = qa['answer_action']

        
        # for question-to-answer shortcuts
        if template not in Q2A_stat:
            Q2A_stat[template] = {}
        if q_keyword not in Q2A_stat[template]:
            Q2A_stat[template][q_keyword] = {}
        
        if a_keyword not in Q2A_stat[template][q_keyword]:
            Q2A_stat[template][q_keyword][a_keyword] = 1
        else:
            Q2A_stat[template][q_keyword][a_keyword] += 1
        # for answer biases
        if template not in answer_stat:
            answer_stat[template] = {}
            
        if answer not in answer_stat[template]:
            answer_stat[template][answer] = 1
        else:
            answer_stat[template][answer] += 1
        
        if template not in action_stat:
            action_stat[template] = {}
            
        if act_id not in action_stat[template]:
            action_stat[template][act_id] = 1
        else:
            action_stat[template][act_id] += 1

    return Q2A_stat,answer_stat, action_stat


def shuffle_option(qa):
    for q in qa:
        question_id = int(q['question_id'].split('_')[-1])
        np.random.seed(question_id)
        q['choices'] = list(np.random.permutation(q['choices']))
        for i in range(4):
            q['choices'][i]['choice_id']=i
    return qa

def group_by_vid(QA):
    qa_by_vid = {}
    vid = {}
    for qa in QA:
        if qa['video_id'] not in qa_by_vid:
            qa_by_vid[qa['video_id']] = [qa]
        else:
            qa_by_vid[qa['video_id']].append(qa)
    for v in qa_by_vid:
        vid[v] = len(qa_by_vid[v])
        
    sorted_vid = sorted(vid.items(), key = lambda x:(x[1],x[0]),reverse=True)
    
    return qa_by_vid, [ id[0] for id in sorted_vid ]

def entropy(statistic):
    num = [statistic[key] for key in statistic if statistic[key]!=0]
    total_num = sum(num)
    c = (np.array(num)/float(total_num)).tolist()
    result=-1;
    if(len(c)>0):
        result=0;
    for x in c:
        result+=(-x)*math.log(x,2)
    return result

def variance(statistic):
    num = np.array([statistic[key] for key in statistic if statistic[key]!=0])
    total_num = sum(num)
    return np.std(num)/total_num

def extract_qa_meta(QA):
    extracted_qas = []
    action_answer = ['Interaction_T4','Sequence_T3','Sequence_T4','Prediction_T1','Feasibility_T4','Feasibility_T6']
    for qa in QA:
        temp_qa = {}
        temp_qa['question_id'] = qa['question_id']
        temp_qa['question'] = qa['question']
        temp_qa['answer'] = qa['answer']
        temp_qa['answer_action'] =qa['answer_action'][0]
        temp_qa['video_id'] = qa['video_id']
        temp_qa['question_keyword']=qa['question_keyword'][0]
        temp_qa['answer_keyword']=qa['answer_keyword'][0]
        qtype = qa['question_id'].split('_')[0]
        qtemp = qa['question_id'].split('_')[1]
        if qtype=='Interaction':
            if qtemp=='T4':
                temp_qa['question_keyword'] = ' '.join(qa['question_keyword'])
        if qtype=='Sequence':
            if qtemp=='T1' or qtemp=='T2' or qtemp=='T3' or qtemp=='T4':
                temp_qa['question_keyword'] = ' '.join(qa['question'].split(' ')[5:])
            if qtemp=='T5' or qtemp=='T6':
                temp_qa['question_keyword'] =  ' '.join(qa['question'].split(' ')[7:])
        if qtype=='Prediction':
            if qtemp=='T1':
                temp_qa['question_keyword'] = 'Prediction_T1'
        if qtype=='Feasibility' :
            if qtemp=='T4':
                temp_qa['question_keyword'] = 'Feasibility_T4'
            if qtemp=='T5':
                temp_qa['question_keyword']= qa['question_keyword'][1]
            if qtemp=='T6':
                temp_qa['question_keyword'] = ' '.join(qa['question'].split(' ')[9:])
        if qtype+'_'+qtemp in action_answer:
            temp_qa['answer_keyword'] = qa['answer']
        extracted_qas.append(temp_qa)

    return extracted_qas

def smooth_sample(ans_stat_, start_filter=0.25, smooth_ratio=0.95, dtype='answer',print_details=False,act_map=None):
    # do_filter_thred: if the answer ratio do not reach the threshold, do not down-sample to keep variaties
    ans_stat= copy.deepcopy(ans_stat_)
    sorted_ans = [item[0] for item in sorted(ans_stat.items(),key=lambda x:(x[1],x[0]))]
    sorted_ans_num = [item[1] for item in sorted(ans_stat.items(),key=lambda x:(x[1],x[0]))]

    start_filter_index = int(len(sorted_ans_num)*(1-start_filter))
    # print(ans_stat_)
    sample_record = {}

    total_num = sum([ans_stat[ans] for ans in sorted_ans])
    # print(total_num)
    sorted_ans_num_ = copy.deepcopy(sorted_ans_num)
        
    for i in range(start_filter_index, len(sorted_ans)-1):

        rest_num = max(int(smooth_ratio*sorted_ans_num[i]),1)
        sample_num = int(max(0,sorted_ans_num[i+1]-rest_num))
        sample_record[sorted_ans[i+1]] = sample_num
        sorted_ans_num[i+1] = sorted_ans_num[i+1] - sample_num
        
    if print_details and act_map is not None:
        for i in range(len(sorted_ans)):
            if dtype == 'action':
                print(act_map[sorted_ans[i]],sorted_ans_num_[i],(sorted_ans_num_[i]/total_num))
            else:
                print(sorted_ans[i],sorted_ans_num_[i],(sorted_ans_num_[i]/total_num))
        print('------------------')    
        total_num2 = sum(sorted_ans_num)
        for i in range(len(sorted_ans)):
            if dtype == 'action':
                print(act_map[sorted_ans[i]],sorted_ans_num[i],(sorted_ans_num[i]/total_num2))
            else:
                print(sorted_ans[i],sorted_ans_num[i],(sorted_ans_num[i]/total_num2))

    total_num2 = sum(sorted_ans_num)
    #print('Rest',total_num2/total_num)
        
    return sample_record

def get_sample_num(QA,thresh,ratio,qtype,dtype,stype='answer'):
    
    _, answer_stat, action_stat = static_distribution(QA)
    #print(action_stat)
    sample_recorder = {}
    if stype=='action':
        sample_stat = action_stat
    elif stype=='answer':
        sample_stat = answer_stat
    for temp in sample_stat:
        stat = sample_stat[temp]
        sample_recorder[temp] = smooth_sample(stat,thresh,ratio)  

    return sample_recorder


def group_by_act(QA):
    result = {}
    for qa in QA:
        act = qa['answer_action']
        ans = qa['answer']
        if ans not in result:
            result[ans] = {}
        if act not in result[ans]:
            result[ans][act]=1
        else:
            result[ans][act]+=1
    return result

# [Template1,Template2,......]
# Template1: [Answer1,Answer2,......]
# Answer1: [Action1,Action2,......]
def sorted_by_temp_ans_act(QA,ans_act_mapping,qtype):
    sorted_qa = []
    TEMPS = {'Interaction':{'T1':[],'T2':[],'T3':[],'T4':[]},
         'Sequence':{'T1':[],'T2':[],'T3':[],'T4':[],'T5':[],'T6':[]},
         'Prediction':{'T1':[],'T2':[],'T3':[],'T4':[]},
         'Feasibility':{'T1':[],'T2':[],'T3':[],'T4':[],'T5':[],'T6':[]}}
    for qa in QA:
        temp = qa['question_id'].split('_')[1]
        TEMPS[qtype][temp].append(qa)
    for temp in TEMPS[qtype]:
        split_by_ans = {}
        for qa in TEMPS[qtype][temp]:
            ans = qa['answer']
            if ans not in split_by_ans:
                split_by_ans[ans] = [qa]
            else:
                split_by_ans[ans].append(qa)
        #print(split_by_ans)
        for ans in split_by_ans:
            qas = split_by_ans[ans]
            acts = [ [qa['answer_action'],i] for i, qa in enumerate(qas)]
            #print(acts)
            if ans in ans_act_mapping:
                #print('here')
                sorted_act = copy.deepcopy(sorted(ans_act_mapping[ans].items(), key=lambda x:x[1], reverse=True))
                sorted_act_id = [item[0] for item in sorted_act]
                #print(sorted_act_id)
                qas_id = [[sorted_act_id.index(act[0]),act[1]] for act in acts]
                #print('ori',qas_id)
                sorted_qas_id = sorted(qas_id,key=lambda x:x[0])
                #print('sorted',sorted_qas_id)
                for id in sorted_qas_id:
                    sorted_qa.append(qas[id[1]])
            else:
                sorted_qa.extend(qas)
    return sorted_qa

def debiasing(QA,thresh,ratio,qtype,dtype,stype):
    filtered_qa = []
    sample_recorder = get_sample_num(QA,thresh,ratio,qtype,dtype,stype)
    ans_act_mapping = group_by_act(QA)
    qa_by_vid, sorted_vid = group_by_vid(QA)
    temp_sample_count = 0
    #print(sample_recorder)
    for vid in sorted_vid:
        qa_in_vid = qa_by_vid[vid]
        qa_in_vid = sorted_by_temp_ans_act(qa_in_vid,ans_act_mapping,qtype)
        for qa in qa_in_vid:
            if stype=='answer':
                #print('here1')
                ans = qa['answer']
                temp = qa['question_id'].split('_')[1]
                if ans in sample_recorder[temp]:
                    if sample_recorder[temp][ans]>0:
                        sample_recorder[temp][ans]-=1
                        temp_sample_count+=1
                    else:
                        filtered_qa.append(qa)
                else:
                    filtered_qa.append(qa)
        
            elif stype=='action':
                ans = qa['answer_action']
                temp = qa['question_id'].split('_')[1]
                if ans in sample_recorder[temp]:
                    if sample_recorder[temp][ans]>0:
                        sample_recorder[temp][ans]-=1
                        temp_sample_count+=1
                        continue
                    else:
                        filtered_qa.append(qa)
                else:
                    filtered_qa.append(qa)
    return filtered_qa

def breaking_shortcuts(QA,mode):

    Q2A_stat, answer_stat, action_stat = static_distribution(QA)
    filtered_qa = []
    sample_recorder = {}
    #print(len(QA))
    for qtemp in Q2A_stat:
        if qtemp not in sample_recorder:
            sample_recorder[qtemp] = {}
        que_ans = Q2A_stat[qtemp]
        for qkey in que_ans:
            if qkey not in sample_recorder[qtemp]:
                sample_recorder[qtemp][qkey] = {}
            
            ans = que_ans[qkey]
            if len(ans.keys())==1:
                sample_recorder[qtemp][qkey][list(ans.keys())[0]] = ans[list(ans.keys())[0]]
            else:
                total_num = sum([item[1] for item in ans.items()])
                sorted_ans = sorted(ans.items(),key=lambda x:x[1],reverse=True)
          
    qa_by_vid, sorted_vid = group_by_vid(QA)
    temp_sample_count = 0
    for vid in sorted_vid:
        qa_in_vid = qa_by_vid[vid]
        for qa in qa_in_vid:
            qtype = qa['question_id'].split('_')[0]
            qtemp = qa['question_id'].split('_')[1]
            q_keyword = qa['question_keyword']
            a_keyword = qa['answer_keyword']
            if q_keyword in sample_recorder[qtemp]:
                if a_keyword in sample_recorder[qtemp][q_keyword]:
                    if sample_recorder[qtemp][q_keyword][a_keyword]>0:
                        sample_recorder[qtemp][q_keyword][a_keyword]-=1
                        continue
                    else:
                        filtered_qa.append(qa)
                else:
                    filtered_qa.append(qa)
                        
            else:
                filtered_qa.append(qa)
    return filtered_qa