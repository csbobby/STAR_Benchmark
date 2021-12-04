"""
Run symbolic reasoning on multiple-choice questions
"""
import os
import json
import argparse
import copy
from tqdm import tqdm
from executor import Executor
from utils import *
import numpy as np
np.random.seed(626)

def split_by_qtype(QA,return_dict=True):
    qa_dict = {'Interaction':[],'Sequence':[],'Prediction':[],'Feasibility':[]}
    for qa in QA:
        qa_dict[qa['question_id'].split('_')[0]].append(qa)
    if return_dict:
        return qa_dict
    return qa_dict['Interaction'], qa_dict['Sequence'], qa_dict['Prediction'], qa_dict['Feasibility']

def execute_program(all_qa,debug=False):
    correct, qa_num = 0, len(all_qa)
    if qa_num==0:
        return 0
    flag = ['Wrong','Correct']
    pbar = tqdm(range(qa_num))
    for ind in pbar:
        qa = all_qa[ind]
        situations = qa['situations']
        exe = Executor(situations)
        q_pg = qa['question_program']
        count = 0        
        for choice in qa['choices']:
            c_pg =  choice['choice_program']
            full_pg = q_pg + c_pg

            pred = exe.run(full_pg, debug)
            if pred == flag[int(choice['choice']==qa['answer'])]:
                count += 1
        if count == len(qa['choices']):
            correct+=1        
        pbar.set_description('Acc {:f}'.format(round(float(correct)*100/qa_num,2)))

    return round(float((correct)*100/qa_num),2)

def situation_reasoning(args):
    qa = json.load(open(args.qa_dir))
    qa_dict = split_by_qtype(qa)

    print('----------Start Reasoning----------')
    for qtype in qa_dict:
        print('----------{}----------'.format(qtype))
        acc = execute_program(qa_dict[qtype])
        print(qtype,'Acc:',acc)
    print('----------End Reasoning----------')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Situation Reasoning")
    parser.add_argument("--qa_dir", default="STAR/STAR_val.json")
    args = parser.parse_args()
    situation_reasoning(args)