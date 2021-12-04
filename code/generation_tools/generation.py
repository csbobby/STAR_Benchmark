# -----------------------------------------------------
# STAR Genetarion 
# -----------------------------------------------------
from generation_tools.utils.nlp import * 
import random
import numpy as np
import copy
import json
RANDOM_SEED = 626
np.random.seed(RANDOM_SEED)

def interaction_t1(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in action_list:
        choices = []
        action_start_time,action_end_time = round(float(each_action[1]), 2),round(float(each_action[2]), 2)
        question_verb = QAG.query_verb(each_action)
        # --------------------------
        # Filter action in sth. + prep. 
        if each_action[0] in QAG.actions_with_something_id:
            continue
        same_verb_actions = QAG.filter_actions_with_verb(action_list,question_verb)
        overlap_actions = QAG.filter_actions_by_time(same_verb_actions,mode='overlap',start=action_start_time,end=action_end_time)
        objs = QAG.filter_objs_in_actions(overlap_actions)
        if QAG.unique(objs):
            # --------------------------
            # QA Generation
            question_words, answer_words={'verbparti':question_verb}, {'obj':objs[0]}
            question, question_program = QAG.generate_question(qtype,temp,question_words)
            correct_answer, op_pro,target_obj = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj')
            choices.append({'choice_id': 0, 'choice': correct_answer,'choice_program':op_pro})
            if generate_option:
                # --------------------------
                # Conflict Option Generation
                question_words, exist_options={'verb':question_verb}, [correct_answer]
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj')
                choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, exist_options, q_key = [target_obj,conflict_obj], [correct_answer,conflict_option], question_verb.split(" ")[0]
                fre_option, op_pro,fre_obj = QAG.generate_frequent_option(qtype,temp,q_key,target_obj,conflict_obj,question_words,exist_words,exist_options,template_type='obj')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words, exist_options = [target_obj,conflict_obj,fre_obj], [correct_answer,conflict_option,fre_option]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
            # --------------------------
            # Situation Graphs
            anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=action_end_time)
            if len(anno_actions)==0:
                continue
            situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            # Filter QA with invalid situations
            if not QAG.pass_situation_graph_checking(situations,each_action[0]):
                continue
            # Filter short videos QA
            if (action_end_time-action_start_time)<QAG.min_video_len:
                continue
            # --------------------------
            QAList.append({'question_id': ('_').join([template_id, str(question_index)]),'video_id': video_id, 'start': action_start_time,'end': action_end_time,'answer_action':[each_action[0]],
                'question_keyword': [question_verb.split(" ")[0]],'answer_keyword': [target_obj],'question': question, 'answer': correct_answer,
                'question_program': question_program, 'choices': choices, 'situations':situations,'anno_actions': anno_actions})
            question_index = question_index + 1
    return QAList, question_index, Question, Answer

def interaction_t2(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in action_list:
        choices = []
        question_obj = QAG.query_obj(each_action)
        action_start_time,action_end_time = round(float(each_action[1]), 2),round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue    
        same_obj_actions = QAG.filter_actions_with_obj(action_list,question_obj)
        overlap_actions = QAG.filter_actions_by_time(same_obj_actions,mode='overlap',start=action_start_time,end=action_end_time)
        verbs = QAG.filter_verbs_in_actions(overlap_actions)
        if QAG.unique(verbs):
            # --------------------------
            # QA Generation
            first_action = QAG.query_earliest_action(overlap_actions) 
            target_verb = QAG.query_verb(first_action)
            question_words, answer_words = {'obj':question_obj}, {'verb':target_verb}
            question, question_program = QAG.generate_question(qtype,temp,question_words)
            correct_answer, op_pro, target_verb = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verbpast')
            choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
            if generate_option:
            # --------------------------
            # Conflict Option Generation
                exist_options = [correct_answer]
                conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='verbpast')
                choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, exist_options, q_key = [target_verb,conflict_verb], [correct_answer,conflict_option], question_obj
                fre_option, op_pro,fre_verb = QAG.generate_frequent_option(qtype,temp,q_key,target_verb,conflict_verb,question_words,exist_words,exist_options,template_type='verbpast')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words, exist_options = [target_verb,conflict_verb,fre_verb]+['hold'], [correct_answer,conflict_option,fre_option]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='verbpast')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
            # --------------------------
            # Annotation
            anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=action_end_time)
            if len(anno_actions)==0:
                continue
            situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            # Filter invalid situations
            if not QAG.pass_situation_graph_checking(situations,each_action[0]):
                continue
            # Filter short videos
            if (action_end_time-action_start_time)<QAG.min_video_len:
                continue
            QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': round(action_start_time, 2),'end': round(action_end_time, 2),
                'answer_action':[each_action[0]],'question_keyword':[question_obj],'answer_keyword':[target_verb.split(" ")[0]],'question': question, 'answer': correct_answer,
                'question_program':question_program , 'choices': choices,'situations':situations,'anno_actions':anno_actions})
            question_index = question_index + 1

    return QAList, question_index, Question, Answer


def interaction_t3(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    memory, avo_rel = [], ['sitting_on','eating','wearing']
    graphs = QAG.scenegraphs(video_id)
    contacting_graphs = QAG.filter_scenegraph_by_reltype(graphs,'contacting')
    if contacting_graphs is None:
        return QAList, question_index, Question, Answer
    for frame in contacting_graphs:
        graph = contacting_graphs[frame]
        happening_actions = QAG.filter_actions_with_frame(action_list,frame,video_id)
        current_time = QAG.query_time(frame,video_id)
        for pair in graph:
            obj = QAG.query_obj(pair)
            rel = QAG.query_rel(pair)
            obj_related_actions = QAG.filter_actions_with_obj(happening_actions,obj)
            if rel in avo_rel or obj_related_actions is None:
                continue
            if QAG.unique(obj_related_actions):
                action = obj_related_actions[0]
                choices = []
                action_start_time,action_end_time = round(float(action[1]),2),round(float(action[2]), 2)
                if action[0] in QAG.actions_with_something_id:
                    continue
                if (current_time - action_start_time) / (action_end_time - action_start_time) > 0.25:
                    continue
                verb = QAG.query_verb(action)
                if set([obj,rel,verb]) in memory:
                        continue
                memory.append(set([obj,rel,verb]))

                question_words, answer_words={'obj':obj,'conrel':rel}, {'verb':verb,'obj':obj}
                question, question_program = QAG.generate_question(qtype,temp,question_words)
                correct_answer, op_pro, target_verb = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verbpast_obj')
                choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

                if generate_option:
                    # --------------------------
                    # Conflict/Compsitional Option Generation
                    answer_words, exist_options = {'verb':[verb,'hold'],'obj':obj}, [correct_answer]
                    conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type1')
                    choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
                    # --------------------------
                    # Frequent Option Generation
                    exist_words, exist_options, q_key = [target_verb,conflict_verb]+['hold'], [correct_answer,conflict_option], obj
                    fre_option, op_pro,fre_verb = QAG.generate_frequent_option(qtype,temp,q_key,target_verb,conflict_verb,question_words,exist_words,exist_options,template_type='fix-type1')
                    choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                    # --------------------------
                    # Random Option Generation
                    exist_words, exist_options = [target_verb,conflict_verb,fre_verb]+['hold'], [correct_answer,conflict_option,fre_option]
                    random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='fix-type1')
                    choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})                    
                    # --------------------------
                    # SG
                anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=action_end_time)
                if len(anno_actions)==0:
                    continue
                situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
                if not QAG.pass_situation_graph_checking(situations,action[0]):
                    continue
                if (action_end_time-action_start_time)<QAG.min_video_len:
                    continue

                QAList.append({'question_id': ('_').join([template_id, str(question_index)]),'video_id': video_id, 'start': action_start_time,'end': action_end_time, 
                    'answer_action':[action[0]],'question_keyword':[obj],'answer_keyword':[verb.split(" ")[0]],'question': question, 'answer': correct_answer,
                    'question_program': question_program, 'choices': choices, 'situations':situations,'anno_actions':anno_actions})
                question_index = question_index + 1

    return QAList, question_index, Question, Answer 

def interaction_t4(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False): # Check
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    # --------------------------
    # Program
    graphs = QAG.scenegraphs(video_id)
    contacting_graphs = QAG.filter_scenegraph_by_reltype(graphs,'contacting')
    contacting_graphs = QAG.filter_scenegraph_without_rel(contacting_graphs,'',avoid=['not_contacting', 'other_relationship','eating','wearing','holding'])
    if contacting_graphs is None:
        return QAList, question_index, Question, Answer

    for frame1 in contacting_graphs:
        obj_rel_pairs = contacting_graphs[frame1]
        for pair in obj_rel_pairs:
            obj1 = QAG.query_obj(pair)
            rel1 = QAG.query_rel(pair)
            rel_irrelated_graphs = QAG.filter_scenegraph_without_rel(contacting_graphs,rel1)
            obj_irrelated_graphs = QAG.filter_scenegraph_without_obj(rel_irrelated_graphs,obj1)
            if obj_irrelated_graphs is None:
                continue

            for frame2 in obj_irrelated_graphs:
                if not QAG.later(frame2,frame1):
                    continue
                obj_rel_pairs2 = obj_irrelated_graphs[frame2]
                for pair2 in obj_rel_pairs2:
                    obj2,rel2 = QAG.query_obj(pair2), QAG.query_rel(pair2)
                    time1,time2 = QAG.query_time(frame1,video_id), QAG.query_time(frame2,video_id)
                    happening_actions = QAG.filter_actions_by_time(action_list,mode='cover',start=time1,end=time2)
                    obj_irrelated_actions = QAG.filter_actions_without_obj(happening_actions,obj1)
                    obj_irrelated_actions = QAG.filter_actions_without_obj(obj_irrelated_actions,obj2)
                    if obj_irrelated_actions is None:
                        continue
                    if not QAG.unique(happening_actions):
                        continue
                    #for action in happening_actions:
                    action = happening_actions[0]
                    if action[0] in QAG.actions_with_something_id:
                        continue
                    choices = []
                    action_start_time, action_end_time =round(float(action[1]), 2), round(float(action[2]), 2)
                    verb,obj = QAG.query_verb(action), QAG.query_obj(action)
                    if set([obj1,rel1,obj2,rel2,verb,obj]) in memory:
                        continue
                    memory.append(set([obj1,rel1,obj2,rel2,verb,obj]))
                    question_words={'obj1':obj1,'conrel1':rel1,'obj2':obj2,'conrel2':rel2}
                    answer_words = {'verb':verb,'obj':obj}
                    question, question_program = QAG.generate_question(qtype,temp,question_words)
                    correct_answer, op_pro, target_verb = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verbpast_obj')
                    choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

                    if generate_option:
                        # --------------------------
                        # Conflict/Compsitional Option Generation
                        answer_words, exist_options = {'verb':[verb,'hold'],'obj':obj}, [correct_answer]
                        conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type2')
                        choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
                        # --------------------------
                        # Frequent Option Generation
                        question_words, q_key, exist_words, exist_options= {'obj':obj}, obj1 + ' ' + obj2, [target_verb,conflict_verb]+['hold'], [correct_answer,conflict_option]
                        fre_option, op_pro,fre_verb = QAG.generate_frequent_option(qtype,temp,q_key,correct_answer,conflict_option,question_words,exist_words,exist_options,template_type='fix-type2')
                        choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                        # --------------------------
                        # Random Option
                        exist_words, exist_options = [target_verb,conflict_verb,fre_verb]+['hold'], [correct_answer,conflict_option,fre_option]
                        random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='fix-type2')
                        choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})     
                    # --------------------------
                    # SG
                    anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=action_end_time)
                    if len(anno_actions)==0:
                        continue
                    situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
                    if not QAG.pass_situation_graph_checking(situations,action[0]):
                        continue
                    if (action_end_time-action_start_time)<QAG.min_video_len:
                        continue
                    QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
                        'video_id': video_id, 'start': action_start_time,'end': action_end_time,
                        'answer_action':[action[0]],'question_keyword':[obj1,obj2],'answer_keyword':[verb.split(" ")[0],obj],
                        'question': question, 'answer': correct_answer,
                        'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
                    question_index = question_index + 1

    return QAList, question_index, Question, Answer  


def sequence_t1(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        if each_action[0] in QAG.actions_with_something_id:
            continue
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        obj2,verb2 = QAG.query_obj(each_action),QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_start_time)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        if next_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj1,verb1 = QAG.query_obj(next_action), QAG.query_verb(next_action)
        video_end_time = round(float(next_action[2]),2)
        if obj1==obj2 or verb1==verb2:
            continue
        # --------------------------
        # QA Generation
        question_words, answer_words={'obj2':obj2,'verb2past':verb2,'verb1':verb1}, {'obj':obj1}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj',obj_index='1')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

        if generate_option:
            question_hint = Answer.replace('[Obj1]',obj2)
            question_words, answer_words, exist_options ={'verb':verb1}, {'obj':[obj1,obj2]}, [correct_answer,question_hint]
            conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, exist_options, q_key = [obj1,obj2,conflict_obj], [correct_answer,conflict_option], ' '.join(question.split(' ')[5:])
            fre_option, op_pro,fre_obj = QAG.generate_frequent_option(qtype,temp,q_key,obj1,conflict_obj,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
            exist_words, exist_options = [obj1,conflict_obj,fre_obj], [correct_answer,conflict_option,fre_option]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
        # --------------------------
        # Annotation
        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=video_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue

        situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (video_end_time-action_start_time)<QAG.min_video_len:
            continue
        # filter similar
        if len(QAG.filter_actions_with_verb(anno_actions,verb2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter invisible
        if obj2 not in QAG.obj_to_ind or QAG.obj_to_ind[obj2] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': action_start_time,'end': video_end_time,
            'answer_action':[next_action[0]],'question_keyword':[verb1.split(" ")[0],verb2.split(" ")[0],obj2],'answer_keyword':[obj1],
            'question': question, 'answer': correct_answer,
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})

        question_index = question_index + 1

    return QAList, question_index, Question, Answer


def sequence_t2(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False): 
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        if each_action[0] in QAG.actions_with_something_id:
            continue
        correct_answer = None
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        # --------------------------
        # Program
        obj2, verb2 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        before_actions = QAG.filter_actions_by_time(action_list, mode='before', start=action_start_time, end=action_end_time)
        last_action = QAG.query_latest_action(before_actions)

        if last_action is None:
            continue
        if last_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj1,verb1 = QAG.query_obj(last_action), QAG.query_verb(last_action)
        video_start_time = round(float(last_action[1]),2)

        if obj1==obj2 or verb1==verb2:
            continue
        # --------------------------
        # QA Generation
        question_words, answer_words = {'obj2':obj2,'verb2past':verb2,'verb1':verb1}, {'obj':obj1}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj',obj_index='1')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

        if generate_option:
            question_hint = Answer.replace('[Obj1]',obj2)
            question_words, answer_words, exist_options = {'verb':verb1},{'obj':[obj1,obj2]}, [correct_answer,question_hint]
            conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, exist_options, q_key = [obj1,obj2,conflict_obj], [correct_answer,conflict_option], ' '.join(question.split(' ')[5:])
            fre_option, op_pro,fre_obj = QAG.generate_frequent_option(qtype,temp,q_key,obj1,conflict_obj,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words, exist_options = [obj1,conflict_obj,fre_obj], [correct_answer,conflict_option,fre_option]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
        # --------------------------
        # Annotation
        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=video_start_time,end=action_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,video_start_time,action_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (action_end_time-video_start_time)<QAG.min_video_len:
            continue
        # filter similar
        if len(QAG.filter_actions_with_verb(anno_actions,verb2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter invisible
        if obj2 not in QAG.obj_to_ind or QAG.obj_to_ind[obj2] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': video_start_time,'end': action_end_time,
            'answer_action':[last_action[0]],'question_keyword':[verb1.split(" ")[0],verb2.split(" ")[0],obj2],'answer_keyword':[obj1],
            'question': question, 'answer': correct_answer, 
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer


def sequence_t3(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        obj1,verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list, mode='after',start=action_start_time)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        if next_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj2, verb2 = QAG.query_obj(next_action), QAG.query_verb(next_action)
        video_end_time = round(float(next_action[2]),2)
        if obj1==obj2 or verb1==verb2:
            continue
        # --------------------------
        # QA Generation
        question_words, answer_words ={'obj1':obj1,'verb1past':verb1}, {'obj':obj2,'verb':verb2}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj_verbpast',obj_index='2',verb_index='2')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

        if generate_option:
            # --------------------------
            # Conflict Option Generation
            converted_verb = verb1.replace(verb1.split(" ")[0], conjugate(verb1.split(" ")[0], tense = PAST, alias = '3sg')) # took
            question_hint = Answer.replace('[Verb2]ed', converted_verb).replace('[Obj2]',obj1)
            question_words, answer_words, exist_words, exist_options = {}, {}, [], [correct_answer,question_hint]
            conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_options, q_key = [correct_answer, conflict_option, question_hint], ' '.join(question.split(' ')[5:])
            fre_option, op_pro,_ = QAG.generate_frequent_option(qtype,temp,q_key,correct_answer,conflict_option,question_words,exist_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_options = [correct_answer,conflict_option,fre_option,question_hint]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        # Annotation
        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=video_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (video_end_time-action_start_time)<QAG.min_video_len:
            continue
        # filter duplicate
        if len(QAG.filter_actions_with_verb(anno_actions,verb2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter ambiguous
        if QAG.exist_same_start_action(anno_actions) or QAG.exist_same_end_action(anno_actions):
            continue
        # filter invisible
        if obj1 not in QAG.obj_to_ind or QAG.obj_to_ind[obj1] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': action_start_time, 'end': video_end_time, 
            'answer_action':[next_action[0]],'question_keyword':[verb1.split(" ")[0],obj1],'answer_keyword':[verb2.split(" ")[0],obj2],
            'question': question, 'answer': correct_answer, 
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer

def sequence_t4(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        obj1,verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        before_actions = QAG.filter_actions_by_time(action_list, mode='before',start=action_start_time,end=action_end_time)
        last_action = QAG.query_latest_action(before_actions)
        if last_action is None:
            continue
        if last_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj2, verb2 = QAG.query_obj(last_action), QAG.query_verb(last_action)
        video_start_time = round(float(last_action[1]),2)
        if obj1==obj2 or verb1==verb2:
            continue
        # --------------------------
        # QA Generation
        question_words, answer_words = {'obj1':obj1,'verb1past':verb1}, {'obj':obj2,'verb':verb2}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj_verbpast',obj_index='2',verb_index='2')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

        if generate_option:
            # --------------------------
            # Conflict Option Generation
            converted_verb = verb1.replace(verb1.split(" ")[0], conjugate(verb1.split(" ")[0], tense = PAST, alias = '3sg')) # took
            question_hint = Answer.replace('[Verb2]ed', converted_verb).replace('[Obj2]',obj1)
            question_words, answer_words, exist_words, exist_options = {}, {}, [], [correct_answer,question_hint]
            conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            q_key = ' '.join(question.split(' ')[5:])
            exist_options = [correct_answer, conflict_option, question_hint]
            fre_option, op_pro,_ = QAG.generate_frequent_option(qtype,temp,q_key,correct_answer,conflict_option,question_words,exist_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_options = [correct_answer,conflict_option,fre_option,question_hint]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action',obj_index='2',verb_index='2')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        # Annotation
        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=video_start_time,end=action_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,video_start_time,action_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (action_end_time-video_start_time)<QAG.min_video_len:
            continue
        # filter duplicate
        if len(QAG.filter_actions_with_verb(anno_actions,verb2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter ambiguous
        if QAG.exist_same_start_action(anno_actions) or QAG.exist_same_end_action(anno_actions):
            continue
        # filter invisible
        if obj1 not in QAG.obj_to_ind or QAG.obj_to_ind[obj1] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': video_start_time, 'end': action_end_time,
            'answer_action':[last_action[0]],'question_keyword':[verb1.split(" ")[0],obj1],'answer_keyword':[verb2.split(" ")[0],obj2],
            'question': question, 'answer': correct_answer,
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer

def sequence_t5(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        obj1, verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list, mode='after',start=action_start_time)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        if next_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj2, verb2 = QAG.query_obj(next_action), QAG.query_verb(next_action)
        video_end_time = round(float(next_action[2]),2)
        if obj1==obj2 or verb1==verb2:
            continue 
        # --------------------------
        # QA Generation
        question_words, answer_words={'obj1':obj1,'obj2':obj2,'verb1present':verb1}, {'verb':verb2}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verbpast',obj_index='2',verb_index='2')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

        if generate_option:
            converted_verb = verb1.replace(verb1.split(" ")[0], conjugate(verb1.split(" ")[0], tense = PAST, alias = '3sg'))
            question_hint = Answer.replace('[Verb2]ed', converted_verb)
            question_words, answer_words, exist_options ={'obj':obj2}, {'verb':[verb1,verb2]}, [correct_answer,question_hint]
            conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, exist_options, q_key = [verb1,verb2,conflict_verb], [correct_answer,conflict_option,question_hint], ' '.join(question.split(' ')[7:])
            fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp,q_key,verb2.split(' ')[0],conflict_verb.split(' ')[0],question_words,exist_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words = [verb1,verb2,conflict_verb,fre_verb]
            exist_options = [correct_answer,conflict_option,fre_option,question_hint]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=action_start_time,end=video_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (video_end_time-action_start_time)<QAG.min_video_len:
            continue
        # filter duplicate
        if len(QAG.filter_actions_with_obj(anno_actions,obj2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter ambiguous
        if QAG.exist_same_start_action(anno_actions) or QAG.exist_same_end_action(anno_actions):
            continue
        # filter invisible
        if obj1 not in QAG.obj_to_ind or QAG.obj_to_ind[obj1] not in QAG.filter_objs_in_graphs(situations):
            continue
        if obj2 not in QAG.obj_to_ind or QAG.obj_to_ind[obj2] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': action_start_time, 'end': video_end_time,
            'answer_action':[next_action[0]],'question_keyword':[verb1.split(" ")[0],obj1,obj2],'answer_keyword':[verb2.split(" ")[0]],
            'question': question, 'answer': correct_answer, 
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer

def sequence_t6(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        obj1, verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        before_actions = QAG.filter_actions_by_time(action_list,mode='before',start=action_start_time,end=action_end_time)
        last_action = QAG.query_latest_action(before_actions)
        if last_action is None:
            continue
        if last_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        obj2, verb2 = QAG.query_obj(last_action), QAG.query_verb(last_action)
        video_start_time = round(float(last_action[1]),2)
        if obj1==obj2 or verb1==verb2:
            continue
        # --------------------------
        # QA Generation
        question_words, answer_words={'obj1':obj1,'obj2':obj2,'verb1present':verb1}, {'verb':verb2}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verbpast',obj_index='2',verb_index='2')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
        if generate_option:
            question_words, answer_words={'obj':obj2}, {'verb':[verb1,verb2]}
            converted_verb = verb1.replace(verb1.split(" ")[0], conjugate(verb1.split(" ")[0], tense = PAST, alias = '3sg'))
            question_hint = Answer.replace('[Verb2]ed', converted_verb)
            exist_options = [correct_answer,question_hint]
            conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, exist_options, q_key = [verb1,verb2,conflict_verb], [correct_answer,conflict_option,question_hint], ' '.join(question.split(' ')[7:])
            fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp,q_key,verb2.split(' ')[0],conflict_verb.split(' ')[0],question_words,exist_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words = [verb1,verb2,conflict_verb,fre_verb]
            exist_options = [correct_answer,conflict_option,fre_option,question_hint]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='verbpast',verb_index='2')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list,mode='in',start=video_start_time,end=action_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,video_start_time,action_end_time,video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1):
            continue
        if (action_end_time-video_start_time)<QAG.min_video_len:
            continue
        # filter duplicate
        if len(QAG.filter_actions_with_obj(anno_actions,obj2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
            continue
        # filter ambiguous QA
        if QAG.exist_same_start_action(anno_actions) or QAG.exist_same_end_action(anno_actions):
            continue
        # filter invisible
        if obj1 not in QAG.obj_to_ind or QAG.obj_to_ind[obj1] not in QAG.filter_objs_in_graphs(situations):
            continue
        if obj2 not in QAG.obj_to_ind or QAG.obj_to_ind[obj2] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 
            'video_id': video_id, 'start': video_start_time, 'end': action_end_time, 
            'answer_action':[last_action[0]],'question_keyword':[verb1.split(" ")[0],obj1,obj2],'answer_keyword':[verb2.split(" ")[0]],
            'question': question, 'answer': correct_answer,
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer
  
def prediction_t1(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id) 

    for each_action in condition_action_list:
        choices = []
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        now_obj, now_verb = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_end_time)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        if next_action[0] in QAG.actions_with_something_id:
            continue
        obj, verb = QAG.query_obj(next_action), QAG.query_verb(next_action)
        if now_obj==obj or now_verb==verb:
            continue
        next_start_time, next_end_time = round(float(next_action[1]),2), round(float(next_action[2]),2)
        video_end_time = next_start_time + (next_end_time - next_start_time) / 4.0
        action_graphs = QAG.filter_scenegraph_by_time(graphs, video_id,start=action_start_time,end=video_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        # ensure visible
        if QAG.obj_to_ind[obj] not in exists_objs:
            continue
        if set([verb,obj]) in memory:
            continue
        memory.append(set([verb,obj]))
        question_words, answer_words = {}, {'obj':obj,'verb':verb}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='obj_verb')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
        if generate_option:
        # --------------------------
        # Conflict Option Generation
            previous_actions = QAG.filter_actions_by_time(action_list, mode='after', end=action_start_time)
            previous_actions = QAG.filter_actions_without_verb(previous_actions,verb)
            previous_actions = QAG.except_(previous_actions,QAG.actions_with_something_id)
            last_action = QAG.query_earliest_action(previous_actions)
            if last_action is None:
                question_words, answer_words, exist_options = {}, {}, [correct_answer]
                conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='action')
            else:
                conflict_obj, conflict_verb = QAG.query_obj(last_action), QAG.query_verb(last_action)
                question_words, answer_words, exist_options = {}, {'obj':conflict_obj,'verb':conflict_verb}, [correct_answer]
                conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type3')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_options, question_words, exist_words, q_key = [correct_answer, conflict_option],{},[], 'Prediction_T1'
            fre_option, op_pro, _ = QAG.generate_frequent_option(qtype,temp,q_key,correct_answer,conflict_option,question_words,exist_words,exist_options,template_type='action')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words, question_words, exist_options = [], {}, [correct_answer,conflict_option,fre_option]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=next_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions, action_start_time, next_end_time, video_id)
        if not QAG.pass_situation_graph_checking(situations,each_action[0],1,frame_num_check=False):
            continue
        if (video_end_time-action_start_time)<QAG.min_video_len:
            continue
        condition_situations = QAG.generate_situation_graphs(anno_actions, action_start_time, video_end_time, video_id)
        if not QAG.pass_situation_graph_checking(condition_situations,each_action[0],action_check=False,min_frame_num=8):
            continue
        # filter duplicate
        dup_actions = QAG.filter_actions_with_obj(anno_actions,obj)
        dup_actions = QAG.filter_actions_with_verb(dup_actions,verb)
        if len(dup_actions)!=1:
            continue
        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,'end': video_end_time, 
            'answer_action':[next_action[0]],'question_keyword':['pre'],'answer_keyword':[verb.split(" ")[0],obj],
            'question': question, 'answer': correct_answer,'question_program': question_program, 'situations':situations,'choices': choices,'anno_actions':anno_actions})

        question_index = question_index + 1

    return QAList, question_index, Question, Answer


def prediction_t2(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        choices = []
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        now_obj, now_verb = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list, mode='after', start=action_end_time)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        if next_action[0] in QAG.actions_with_something_id:
            continue
        obj, verb = QAG.query_obj(next_action), QAG.query_verb(next_action)
        next_start_time, next_end_time = float(next_action[1]), float(next_action[2])
        video_end_time = next_start_time + (next_end_time - next_start_time)/4.0
        if obj==now_obj or verb==now_verb:
            continue
        action_graphs = QAG.filter_scenegraph_by_time(graphs, video_id, start=action_start_time, end=video_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        if QAG.obj_to_ind[obj] not in exists_objs:
            continue
        if set([obj,verb]) in memory:
            continue
        memory.append(set([obj,verb]))
        question_words, answer_words = {'obj':obj}, {'verb':verb}
        question, question_program = QAG.generate_question(qtype,temp,question_words)
        correct_answer, op_pro, _ = QAG.generate_correct_answer(qtype,temp,answer_words, template_type='verb')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
        if generate_option:
        # --------------------------
        # Conflict/Compsitional Option Generation
            before_actions = QAG.filter_actions_by_time(action_list,mode='after',end=action_start_time)
            obj_relavant_actions = QAG.filter_actions_with_obj(before_actions,obj)
            different_verb_actions = QAG.filter_actions_without_verb(obj_relavant_actions,verb)
            different_verb_actions = QAG.except_(different_verb_actions,QAG.actions_with_something_id)
            fisrt_verb_changed_action = QAG.query_earliest_action(different_verb_actions)

            if fisrt_verb_changed_action is None:
                question_words, answer_words, exist_options = {'obj':obj}, {'verb':verb}, [correct_answer]
                conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='verb')
            else:
                conflict_verb = QAG.query_verb(fisrt_verb_changed_action)
                question_words, answer_words, exist_options = {}, {'verb':conflict_verb}, [correct_answer]
                conflict_option, op_pro, conflict_verb = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type4')
            choices.append({'choice_id': 1, 'choice': conflict_option,'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, question_words, exist_options, q_key = [verb,conflict_verb], {'obj':obj}, [correct_answer,conflict_option], obj
            fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp, q_key, verb.split(' ')[0], conflict_verb.split(' ')[0], question_words, exist_words, exist_options, template_type='verb')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words, question_words, exist_options = [verb,fre_verb,conflict_verb], {'obj':obj}, [correct_answer,conflict_option,fre_option]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='verb')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=next_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        if QAG.exist_inner_actions(next_action,anno_actions) or QAG.exist_same_end_action(anno_actions):
            continue
        situations = QAG.generate_situation_graphs(anno_actions, action_start_time, next_end_time, video_id)
        if situations is None:
            continue
        extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
        if len(extracted_actions)<=1:
            continue
        if each_action[0] not in extracted_actions:
            continue
        condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if condition_situations is None:
            continue
        if len(condition_situations.keys())<QAG.min_keyframes:
            continue
        dup_actions = QAG.filter_actions_with_obj(anno_actions,obj)
        if len(dup_actions)!=1:
            continue
        if (obj not in QAG.obj_to_ind) or (QAG.obj_to_ind[obj] not in QAG.filter_objs_in_graphs(situations)):
            continue
        if (video_end_time-action_start_time)<QAG.min_video_len:
            continue
        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,'end': video_end_time, 
            'answer_action':[next_action[0]],'question_keyword':[obj],'answer_keyword':[verb.split(" ")[0]], 'question': question, 'answer': correct_answer,
            'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer
               

def prediction_t3(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False): 
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        choices = []
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        after_actions = QAG.filter_actions_by_time(action_list, mode='after', start=action_end_time)
        after_actions = QAG.except_(after_actions,QAG.actions_with_something_id)
        next_action = QAG.query_earliest_action(after_actions)
        if next_action is None:
            continue
        now_obj, now_verb = QAG.query_obj(each_action), QAG.query_verb(each_action)
        obj, verb = QAG.query_obj(next_action), QAG.query_verb(next_action)
        next_start_time, next_end_time = float(next_action[1]), float(next_action[2])
        video_end_time = next_start_time + (next_start_time - next_end_time)/4.0
        if now_obj==obj or now_verb==verb:
            continue
        action_graphs = QAG.filter_scenegraph_by_time(graphs, video_id, start=action_start_time, end=video_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        if QAG.obj_to_ind[obj] not in exists_objs:
            continue
        if set([verb,obj]) in memory:
            continue
        memory.append(set([verb,obj]))
        # --------------------------
        # QA Generation
        question_words, answer_words = {'verb':verb}, {'obj':obj}
        question, question_program = QAG.generate_question(qtype, temp, question_words)
        correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
        if generate_option:
            # --------------------------
            # Conflict Option Generation
            before_actions = QAG.filter_actions_by_time(action_list, mode='after', end=action_start_time)
            before_actions = QAG.filter_actions_without_obj(before_actions, obj)
            before_action = QAG.query_earliest_action(before_actions)
            if before_action is None:
                question_words, answer_words, exist_options = {'verb':verb}, {'obj':obj}, [correct_answer]
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj')
            else:
                conflict_obj = QAG.query_obj(before_action)
                question_words, answer_words, exist_options = {}, {'obj':conflict_obj}, []
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type5')
            choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, question_words, exist_options, q_key = [obj,conflict_obj], {'verb':verb}, [correct_answer,conflict_option], verb.split(" ")[0]
            fre_option, op_pro, fre_obj = QAG.generate_frequent_option(qtype,temp, q_key, obj, conflict_obj, question_words, exist_words, exist_options, template_type='obj')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words, question_words, exist_options = [obj,conflict_obj,fre_obj], {'verb':verb}, [correct_answer,conflict_option,fre_option]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=next_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        if QAG.exist_same_start_action(anno_actions):
            continue
        situations = QAG.generate_situation_graphs(anno_actions, action_start_time, next_end_time, video_id)
        if situations is None:
            continue
        condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if condition_situations is None:
            continue
        if len(condition_situations.keys())<QAG.min_keyframes:
            continue
        extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
        if len(extracted_actions)<=1:
            continue
        if each_action[0] not in extracted_actions:
            continue
        # # filter duplicate
        dup_actions = QAG.filter_actions_with_verb(anno_actions,verb)
        if len(dup_actions)!=1:
            continue
        if obj not in QAG.obj_to_ind or QAG.obj_to_ind[obj] not in QAG.filter_objs_in_graphs(situations):
            continue
        if (video_end_time-action_start_time) < QAG.min_video_len:
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,'end': video_end_time,
        'answer_action':[next_action[0]],'question_keyword':[verb.split(" ")[0]],'answer_keyword':[obj],'question': question, 'answer': correct_answer,
        'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer


def prediction_t4(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        if each_action[0] in QAG.actions_with_something_id:
            continue
        choices = []
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        obj1,verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        next_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_end_time)
        next_action = QAG.query_earliest_action(next_actions)
        if next_action is None:
            continue
        verb2, obj2 = QAG.query_verb(next_action), QAG.query_obj(next_action)
        next_start_time, next_end_time = round(float(next_action[1]),2), round(float(next_action[2]),2)
        video_end_time = next_start_time + (next_end_time - next_start_time) / 4.0
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        if QAG.obj_to_ind[obj2] not in exists_objs:
            continue
        if obj1==obj2 or verb2==verb1:
            continue
        # --------------------------
        # QA Generation 
        question_words, answer_words = {'obj1':obj1,'verb1':verb1,'verb2':verb2}, {'obj':obj2}
        question, question_program = QAG.generate_question(qtype, temp, question_words)
        correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj',obj_index='2')
        choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
        if generate_option:
            # --------------------------
            # Conflict/Compositional Option Generation
            question_hint = Answer.replace('[Obj2]',obj1)
            before_actions = QAG.filter_actions_by_time(action_list, mode='after',end = action_start_time)
            before_actions = QAG.filter_actions_without_obj(before_actions,obj1)
            before_actions = QAG.filter_actions_without_obj(before_actions,obj2)
            before_actions = QAG.filter_actions_with_verb(before_actions,verb1)
            before_action = QAG.query_earliest_action(before_actions)
            if before_action is None:
                question_words, answer_words, exist_options = {'verb':verb2}, {'obj':[obj1,obj2]}, [correct_answer,question_hint]
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj',obj_index='2')
            else:
                conflict_obj = QAG.query_obj(before_action)
                question_words, answer_words, exist_options = {}, {'obj':conflict_obj}, []
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type5')
            choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
            # --------------------------
            # Frequent Option Generation
            exist_words, question_words, exist_options, q_key = [obj1,obj2,conflict_obj], {'verb':verb2}, [correct_answer,conflict_option,question_hint], verb2.split(" ")[0]
            fre_option, op_pro, fre_obj = QAG.generate_frequent_option(qtype,temp, q_key, obj2, conflict_obj, question_words, exist_words, exist_options, template_type='obj',obj_index='2')
            choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
            # --------------------------
            # Random Option Generation
            exist_words, question_words, exist_options = [obj1,obj2,conflict_obj,fre_obj], {'verb':verb2}, [correct_answer,conflict_option,fre_option,question_hint]
            random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj',obj_index='2')
            choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

        anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=next_end_time)
        if each_action not in anno_actions:
            anno_actions.append(each_action)
        if len(anno_actions)<=1:
            continue
        situations = QAG.generate_situation_graphs(anno_actions,action_start_time,next_end_time,video_id)
        if situations is None:
            continue
        condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,video_end_time,video_id)
        if condition_situations is None:
            continue
        if len(condition_situations.keys())<QAG.min_keyframes:
            continue
        extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
        if len(extracted_actions)<=1:
            continue
        if each_action[0] not in extracted_actions:
            continue
        if (video_end_time-action_start_time) < QAG.min_video_len:
            continue
        if QAG.exist_same_start_action(anno_actions):
            continue
        dup_actions = QAG.filter_actions_with_verb(anno_actions,verb2)
        if len(dup_actions)!=1:
            continue 
        if obj2 not in QAG.obj_to_ind or QAG.obj_to_ind[obj2] not in QAG.filter_objs_in_graphs(situations):
            continue

        QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time, 'end': video_end_time, 
            'answer_action':[next_action[0]],'question_keyword':[verb2.split(" ")[0],verb1.split(" ")[0],obj1],'answer_keyword':[obj2],
            'question': question, 'answer': correct_answer, 'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
        question_index = question_index + 1

    return QAList, question_index, Question, Answer

def feasibility_t1(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)
    for each_action in action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        after_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_end_time)
        if after_actions is None:
            continue 
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        duplicate_verb, now_obj = QAG.query_verb(each_action), QAG.query_obj(each_action)
        fea_actions = QAG.filter_actions_with_verb(after_actions,duplicate_verb)
        fea_actions = QAG.filter_actions_without_obj(fea_actions,now_obj)
        fea_actions = QAG.filter_actions_by_time(fea_actions,mode='after',start=action_end_time)
        if fea_actions is None:
            continue
        for fea_action in fea_actions:
            choices = []
            fea_obj = QAG.query_obj(fea_action)
            fea_action_end = float(fea_action[2])
            if QAG.obj_to_ind[fea_obj] not in exists_objs:
                continue
            if now_obj==fea_obj:
                continue
            if set([duplicate_verb,fea_obj,action_start_time,action_end_time]) in memory:
                continue
            memory.append(set([duplicate_verb,fea_obj,action_start_time,action_end_time]))
            # --------------------------
            # QA Generation 
            question_words, answer_words = {'verbparti':duplicate_verb}, {'obj':fea_obj}
            question, question_program = QAG.generate_question(qtype, temp, question_words)
            correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj')
            choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})

            if generate_option:
                # --------------------------
                # Conflict Option Generation
                question_words, answer_words, exist_options = {}, {'obj':now_obj}, []
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type5')        
                choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, question_words, exist_options, q_key = [], {}, [correct_answer,conflict_option], duplicate_verb.split(" ")[0]
                fre_option, op_pro, fre_obj = QAG.generate_frequent_option(qtype,temp, q_key, fea_obj, now_obj, question_words, exist_words, exist_options, template_type='fix-type3')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words, question_words, exist_options = [fea_obj,now_obj,fre_obj], {'verb':duplicate_verb}, [correct_answer,conflict_option,fre_option]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='uncompobj')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

            anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=fea_action_end)
            if each_action not in anno_actions:
                anno_actions.append(each_action)
            if len(anno_actions)<=1:
                continue
            if QAG.exist_same_start_action(anno_actions):
                continue
            situations = QAG.generate_situation_graphs(anno_actions,action_start_time,fea_action_end,video_id)
            if situations is None:
                continue
            condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            if condition_situations is None:
                continue
            if len(condition_situations.keys())<QAG.min_keyframes:
                continue
            extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
            if len(extracted_actions)<=1:
                continue
            if each_action[0] not in extracted_actions:
                continue
            if (action_end_time-action_start_time)<QAG.min_video_len:
                continue
            QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,'end': action_end_time,
                'answer_action': [fea_action[0]], 'question_keyword':[duplicate_verb.split(" ")[0]],'answer_keyword':[fea_obj],
                'question': question, 'answer': correct_answer, 'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})

            question_index = question_index + 1

    return QAList, question_index, Question, Answer


def feasibility_t2(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        now_verb, duplicate_obj = QAG.query_verb(each_action), QAG.query_obj(each_action)
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        fea_actions = QAG.filter_actions_with_obj(action_list,duplicate_obj)
        fea_actions = QAG.filter_actions_by_time(fea_actions,mode='after',start=action_end_time)
        if fea_actions is None:
            continue
        for fea_action in fea_actions:
            choices = []
            fea_verb = QAG.query_verb(fea_action)
            fea_action_end = round(float(fea_action[2]),2)
            if fea_verb==now_verb:
                continue
            if set([duplicate_obj,fea_verb,action_start_time,action_end_time]) in memory:
                continue
            memory.append(set([duplicate_obj,fea_verb,action_start_time,action_end_time]))

            question_words, answer_words = {'obj':duplicate_obj}, {'verb':fea_verb,'obj':duplicate_obj}
            question, question_program = QAG.generate_question(qtype, temp, question_words)
            correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj_verb')
            choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
            if generate_option:

                question_words, answer_words, exist_options = {}, {'obj':duplicate_obj,'verb':now_verb}, []
                conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type3')        
                choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, question_words, exist_options, q_key = [fea_verb,now_verb], {'obj':duplicate_obj}, [correct_answer,conflict_option], duplicate_obj
                fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp, q_key, fea_verb.split(" ")[0], now_verb.split(" ")[0], question_words, exist_words, exist_options, template_type='uncompverb')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words, question_words, exist_options = [fea_verb,now_verb,fre_verb], {'obj':duplicate_obj}, [correct_answer,conflict_option,fre_option]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='uncompverb')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

            anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=fea_action_end)
            if each_action not in anno_actions:
                anno_actions.append(each_action)
            if len(anno_actions)<=1:
                continue
            if QAG.exist_same_start_action(anno_actions):
                continue
            situations = QAG.generate_situation_graphs(anno_actions,action_start_time,fea_action_end,video_id)
            if situations is None:
                continue
            extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
            if len(extracted_actions)<=1:
                continue
            if each_action[0] not in extracted_actions:
                continue
            condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            if condition_situations is None:
                continue
            if len(condition_situations.keys())<QAG.min_keyframes:
                continue
            if action_end_time - action_start_time < QAG.min_video_len:
                continue

            QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,
            'end': action_end_time, 'answer_action': [fea_action[0]], 'question_keyword':[duplicate_obj],'answer_keyword':[fea_verb.split(' ')[0],duplicate_obj],
            'question': question, 'answer': correct_answer,'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
            question_index = question_index + 1

    return QAList, question_index, Question, Answer


def feasibility_t3(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        if each_action in action_list:
            continue
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)
        spatial_graphs = QAG.filter_spatial_relations_in_graphs(action_graphs)
        start_condition, end_condition = QAG.query_first_graph(spatial_graphs), QAG.query_last_graph(spatial_graphs)
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        next_actions = QAG.filter_actions_by_time(action_list, mode='after', start=action_end_time)
        if next_actions is None:
            continue
        changed_state = QAG.filter_changed_relations(start_condition,end_condition)
        if changed_state is None:
            continue
        for i in range(len(changed_state['rel_pairs'])):
            rel, spatial_rel_ind= changed_state['rel_pairs'][i], changed_state['rel_labels'][i]
            spatial_rel, obj2 = QAG.ind_to_rel[spatial_rel_ind], QAG.ind_to_obj[rel[0]]
            if spatial_rel in QAG.realation_inverse:
                spatial_rel = QAG.realation_inverse[spatial_rel]
            different_obj_actions = QAG.filter_actions_without_obj(next_actions,obj2)
            if different_obj_actions is None:
                continue
            for action in different_obj_actions:
                choices = []
                obj1, verb1 = QAG.query_obj(action), QAG.query_verb(action)
                next_action_start_time, next_action_end_time = round(float(action[1]), 2), round(float(action[2]), 2)
                if QAG.obj_to_ind[obj1] not in exists_objs:
                    continue
                if set([spatial_rel,verb1,obj1,obj2]) in memory:
                    continue
                memory.append(set([spatial_rel,verb1,obj1,obj2]))
                somewhere_list = 'door,table,bed,chair,shelf,closet,cabinet,refrigerator,sofa,coach,window'.split(',')
                if obj2 not in somewhere_list:
                    continue
                # --------------------------
                # QA Generation
                question_words, answer_words = {'sparel':spatial_rel,'obj2':obj2,'verb1parti':verb1}, {'obj':obj1}
                question, question_program = QAG.generate_question(qtype, temp, question_words)
                correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj',obj_index='1')
                choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
                if generate_option:
                    # --------------------------
                    # Conflict Option Generation
                    question_hint = Answer.replace('[Obj1]',obj2)
                    if QAG.obj_to_ind[obj1] in exists_objs:
                        exists_objs.remove(QAG.obj_to_ind[obj1])
                    if QAG.obj_to_ind[obj2] in exists_objs:
                        exists_objs.remove(QAG.obj_to_ind[obj2])
                    # todo
                    if len(exists_objs)==0:
                        conflict_obj = QAG.compositional_obj(verb1,[obj1,obj2])
                        if conflict_obj is None:
                            conflict_obj = QAG.random_obj([obj1,obj2])
                    else:   
                        np.random.seed(question_index)
                        index = np.random.randint(0,len(exists_objs))
                        conflict_obj = QAG.ind_to_obj[exists_objs[index]]

                    question_words, answer_words, exist_options = {}, {'obj':conflict_obj}, [correct_answer,question_hint]
                    conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type6',obj_index='1') 
                    choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                    # --------------------------
                    # Frequent Option Generation
                    exist_words, question_words, exist_options, q_key = [], {}, [correct_answer,conflict_option,question_hint], verb1.split(" ")[0]
                    fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp, q_key, obj1, conflict_obj, question_words, exist_words, exist_options, template_type='fix-type3',obj_index='1')
                    choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                    # --------------------------
                    # Random Option Generation
                    exist_words, question_words, exist_options = [], {}, [correct_answer,conflict_option,fre_option,question_hint]
                    random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action',obj_index='1')
                    choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
                # todo
                anno_actions = [each_action] + next_actions
                situations = QAG.generate_situation_graphs(anno_actions, action_start_time, next_action_end_time, video_id,'extend')
                if situations is None:
                    continue
                condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
                if condition_situations is None:
                    continue
                if len(condition_situations.keys())<QAG.min_keyframes:
                    continue
                extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
                if len(extracted_actions)<=1:
                    continue
                if each_action[0] not in extracted_actions:
                    continue
                if (action_end_time-action_start_time)<QAG.min_video_len:
                    continue
                QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,
                'end': action_end_time, 'answer_action': [action[0]], 'question_keyword':[verb1.split(" ")[0],obj2],'answer_keyword':[obj1],
                'question': question, 'answer': correct_answer, 'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
                question_index = question_index + 1

    return QAList, question_index, Question, Answer

def feasibility_t4(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        if each_action in action_list:
            continue
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)
        spatial_graphs = QAG.filter_spatial_relations_in_graphs(action_graphs)
        start_condition, end_condition = QAG.query_first_graph(spatial_graphs), QAG.query_last_graph(spatial_graphs)
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        next_actions = QAG.filter_actions_by_time(action_list, mode='after', start=action_end_time)
        if next_actions is None:
            continue
        changed_state = QAG.filter_changed_relations(start_condition,end_condition)
        if changed_state is None:
            continue
        for i in range(len(changed_state['rel_pairs'])):
            rel, spatial_rel_ind= changed_state['rel_pairs'][i], changed_state['rel_labels'][i]
            spatial_rel, obj2 = QAG.ind_to_rel[spatial_rel_ind], QAG.ind_to_obj[rel[0]]
            if spatial_rel in QAG.realation_inverse:
                spatial_rel = QAG.realation_inverse[spatial_rel]
            different_obj_actions = QAG.filter_actions_without_obj(next_actions,obj2)
            if different_obj_actions is None:
                continue
            for action in different_obj_actions:
                choices = []
                obj1, verb1 = QAG.query_obj(action), QAG.query_verb(action)
                next_action_start_time, next_action_end_time = round(float(action[1]), 2), round(float(action[2]), 2)
                if QAG.obj_to_ind[obj1] not in exists_objs:
                    continue
                if set([spatial_rel,verb1,obj1,obj2]) in memory:
                    continue
                memory.append(set([spatial_rel,verb1,obj1,obj2]))
                somewhere_list = 'door,table,bed,chair,shelf,closet,cabinet,refrigerator,sofa,coach,window'.split(',')
                if obj2 not in somewhere_list:
                    continue
                question_words, answer_words = {'sparel':spatial_rel,'obj2':obj2}, {'obj':obj1, 'verb':verb1}
                question, question_program = QAG.generate_question(qtype, temp, question_words)
                correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='verb_obj',obj_index='1',verb_index='1')
                choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
                if generate_option:
                    # --------------------------
                    # Conflict Option Generation
                    if QAG.obj_to_ind[obj1] in exists_objs:
                        exists_objs.remove(QAG.obj_to_ind[obj1])
                    if QAG.obj_to_ind[obj2] in exists_objs:
                        exists_objs.remove(QAG.obj_to_ind[obj2])
                    # todo
                    if len(exists_objs)==0:
                        conflict_obj = QAG.compositional_obj(verb1,[obj1,obj2])
                        if conflict_obj is None:
                            conflict_obj = QAG.random_obj([obj1,obj2])
                    else:   
                        np.random.seed(question_index)
                        index = np.random.randint(0,len(exists_objs))
                        conflict_obj = QAG.ind_to_obj[exists_objs[index]]
                    conflict_verb = QAG.compositional_verb(conflict_obj,[verb1])
                    if conflict_verb is None:
                        conflict_verb = QAG.random_verb([verb1])
                    question_words, answer_words, exist_options = {}, {'obj':conflict_obj,'verb':conflict_verb}, [correct_answer]
                    conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type3',obj_index='1',verb_index='1') 
                    choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                    # --------------------------
                    # Frequent Option Generation
                    exist_words, question_words, exist_options, q_key = [], {}, [correct_answer,conflict_option], 'Feasibility_T4'
                    fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp, q_key, correct_answer, conflict_option, question_words, exist_words, exist_options, template_type='action',obj_index='1',verb_index='1')
                    choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                    # --------------------------
                    # Random Option Generation
                    exist_words, question_words, exist_options = [], {}, [correct_answer,conflict_option,fre_option]
                    random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action',obj_index='1',verb_index='1')
                    choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})
                # todo
                anno_actions = [each_action] + next_actions
                situations = QAG.generate_situation_graphs(anno_actions, action_start_time, next_action_end_time, video_id,'extend')
                if situations is None:
                    continue
                condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
                if condition_situations is None:
                    continue
                if len(condition_situations.keys())<QAG.min_keyframes:
                    continue
                extracted_actions = QAG.extract_actions_from_situiation_graphs(situations)
                if len(extracted_actions)<=1:
                    continue
                if each_action[0] not in extracted_actions:
                    continue
                if (action_end_time-action_start_time)<QAG.min_video_len:
                    continue
                QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,
                'end': action_end_time,'question_keyword':[obj2],'answer_keyword':[obj1,verb1.split(' ')[0]],'question': question, 'answer': correct_answer, 'answer_action': [action[0]], 
                'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})
                question_index = question_index + 1

    return QAList, question_index, Question, Answer


def feasibility_t5(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        # --------------------------
        # Program
        obj2, verb2 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_end_time)
        if after_actions is None:
            continue
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        for action in after_actions:
            if action[0] in QAG.actions_with_something_id:
                continue
            choices = []
            obj1, verb1 = QAG.query_obj(action), QAG.query_verb(action)
            fea_action_end = round(float(action[2]), 2)
            if QAG.obj_to_ind[obj1] not in exists_objs:
                continue
            if obj1==obj2 or verb1==verb2:
                continue
            if set([verb1,verb2,obj1,obj2]) in memory:
                continue
            memory.append(set([verb1,verb2,obj1,obj2]))
            # --------------------------
            # QA Generation
            question_words, answer_words = {'verb2present':verb2,'obj2':obj2,'verb1':verb1}, {'obj':obj1}
            question, question_program = QAG.generate_question(qtype, temp, question_words)
            correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj',obj_index='1',verb_index='1')
            choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
            if generate_option:
                question_hint = Answer.replace('[Obj1]',obj2)
                # --------------------------
                # Conflict Option Generation
                question_words, answer_words, exist_options = {'verb':verb1}, {'obj':[obj2,obj1]}, [correct_answer,question_hint]
                conflict_option, op_pro, conflict_obj = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='obj',obj_index='1')        
                choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, exist_options, q_key = [obj1,obj2,conflict_obj], [correct_answer,conflict_option,question_hint], verb1.split(" ")[0]
                fre_option, op_pro,fre_obj = QAG.generate_frequent_option(qtype,temp,q_key,obj1,conflict_obj,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words = exists_objs+[fre_obj,obj1,obj2,conflict_obj]
                exist_options = [correct_answer,conflict_option,fre_option,question_hint]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='obj',obj_index='1')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

            anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=fea_action_end)
            if each_action not in anno_actions:
                anno_actions.append(each_action)
            if len(anno_actions)<=1:
                continue
            situations = QAG.generate_situation_graphs(anno_actions, action_start_time, fea_action_end, video_id)
            if not QAG.pass_situation_graph_checking(situations,each_action[0],1,frame_num_check=False):
                continue
            if (action_end_time-action_start_time)<QAG.min_video_len:
                continue
            condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            if not QAG.pass_situation_graph_checking(condition_situations,each_action[0],action_check=False):
                continue
            if len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
                continue

            QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,
            'end': action_end_time, 'answer_action': [action[0]], 'question_keyword':[verb2.split(" ")[0],verb1.split(" ")[0],obj2],'answer_keyword':[obj1],
            'question': question, 'answer': correct_answer,'question_program': question_program, 'choices': choices,'situations':situations,'anno_actions':anno_actions})

            question_index = question_index + 1

    return QAList, question_index, Question, Answer


def feasibility_t6(QAG, template_id, QAList, action_list, condition_action_list, question_index, video_id, generate_option=False):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = QAG.qa_template_dict[qtype][temp]['question'], QAG.qa_template_dict[qtype][temp]['answer']
    memory = []
    graphs = QAG.extend_scenegraphs(video_id)

    for each_action in condition_action_list:
        action_start_time, action_end_time = round(float(each_action[1]), 2), round(float(each_action[2]), 2)
        if each_action[0] in QAG.actions_with_something_id:
            continue
        # --------------------------
        # Program
        obj1, verb1 = QAG.query_obj(each_action), QAG.query_verb(each_action)
        after_actions = QAG.filter_actions_by_time(action_list,mode='after',start=action_end_time)
        if after_actions is None:
            continue
        action_graphs = QAG.filter_scenegraph_by_time(graphs,video_id,start=action_start_time,end=action_end_time)        
        exists_objs = QAG.filter_objs_in_graphs(action_graphs)
        if exists_objs is None:
            continue
        for action in after_actions:
            if action[0] in QAG.actions_with_something_id:
                continue
            choices = []
            obj2, verb2 = QAG.query_obj(action), QAG.query_verb(action)
            fea_action_end = round(float(action[2]), 2)
            if QAG.obj_to_ind[obj2] not in exists_objs:
                continue
            if obj1==obj2 or verb1==verb2:
                continue
            if set([verb1,verb2,obj1,obj2]) in memory:
                continue
            memory.append(set([verb1,verb2,obj1,obj2]))
            # --------------------------
            # QA Generation
            question_words, answer_words = {'verb1present':verb1,'obj1':obj1}, {'obj':obj2,'verb':verb2}
            question, question_program = QAG.generate_question(qtype, temp, question_words)
            correct_answer, op_pro, target_obj = QAG.generate_correct_answer(qtype,temp, answer_words, template_type='obj_verb',obj_index='2',verb_index='2')
            choices.append({'choice_id': 0, 'choice': correct_answer, 'choice_program':op_pro})
            if generate_option:
                # --------------------------
                # QA Generation
                question_hint = Answer.replace('[Obj2]',obj1).replace('[Verb2]',verb1)
                before_actions = QAG.filter_actions_by_time(action_list,mode='after',end=action_end_time)
                before_actions = QAG.filter_actions_without_obj(before_actions,obj1)
                before_actions = QAG.filter_actions_without_obj(before_actions,obj2)
                before_actions = QAG.filter_actions_without_verb(before_actions,verb1)
                before_actions = QAG.filter_actions_without_verb(before_actions,verb2)
                last_action = QAG.query_latest_action(before_actions)
                # todo
                if last_action is None or QAG.obj_to_ind[QAG.query_obj(last_action)] not in exists_objs:
                    exists_objs.remove(QAG.obj_to_ind[obj2])
                    if len(exists_objs) >= 1:
                        np.random.seed(question_index)
                        conflict_obj = QAG.ind_to_obj[exists_objs[np.random.randint(0,len(exists_objs))]]
                    else:
                        conflict_obj = QAG.compositional_obj(verb2,[obj2,obj1])
                else:
                    conflict_obj = QAG.query_obj(last_action)

                conflict_verb = QAG.compositional_verb(conflict_obj,[verb2])
                if conflict_verb is None:
                    conflict_obj = obj2
                    conflict_verb = QAG.compositional_verb(conflict_obj,[verb2])
                question_words, answer_words, exist_options = {}, {'obj':conflict_obj,'verb':conflict_verb}, [correct_answer,question_hint]
                conflict_option, op_pro, _ = QAG.generate_conflict_option(qtype,temp,question_words,answer_words,exist_options,template_type='fix-type7',obj_index='2',verb_index='2') 
                choices.append({'choice_id': 1, 'choice': conflict_option, 'choice_program':op_pro})
                # --------------------------
                # Frequent Option Generation
                exist_words, question_words, exist_options, q_key = [], {}, [correct_answer,conflict_option,question_hint], ' '.join(question.split(' ')[9:])
                fre_option, op_pro, fre_verb = QAG.generate_frequent_option(qtype,temp, q_key, correct_answer, conflict_option, question_words, exist_words, exist_options, template_type='action',obj_index='2',verb_index='2')
                choices.append({'choice_id': 2, 'choice': fre_option, 'choice_program':op_pro})
                # --------------------------
                # Random Option Generation
                exist_words, question_words, exist_options = [], {}, [correct_answer,conflict_option,fre_option,question_hint]
                random_option, op_pro = QAG.generate_random_option(qtype,temp,question_words,exist_words,exist_options,template_type='action',obj_index='2',verb_index='2')
                choices.append({'choice_id': 3, 'choice': random_option, 'choice_program':op_pro})

            anno_actions = QAG.filter_actions_by_time(action_list, mode='in', start=action_start_time, end=fea_action_end)
            if each_action not in anno_actions:
                anno_actions.append(each_action)
            if len(anno_actions)<=1:
                continue
            situations = QAG.generate_situation_graphs(anno_actions, action_start_time, fea_action_end, video_id)
            if not QAG.pass_situation_graph_checking(situations,each_action[0],1,frame_num_check=False):
                continue
            condition_situations = QAG.generate_situation_graphs(anno_actions,action_start_time,action_end_time,video_id)
            if not QAG.pass_situation_graph_checking(condition_situations,each_action[0],action_check=False):
                continue
            # filter duplicate
            if len(QAG.filter_actions_with_verb(anno_actions,verb2))!=1 or len(QAG.filter_actions_with_verb(anno_actions,verb1))!=1:
                continue
            if (action_end_time-action_start_time)<QAG.min_video_len:
                continue
            QAList.append({'question_id': ('_').join([template_id, str(question_index)]), 'video_id': video_id, 'start': action_start_time,'answer_action': [action[0]], 
                           'end': action_end_time, 'question': question, 'answer': correct_answer,'question_keyword':[verb1.split(" ")[0],obj1],'answer_keyword':[verb2.split(" ")[0],obj2],
                           'question_program': question_program, 'choices': choices, 'situations':situations,'anno_actions':anno_actions})
            question_index = question_index + 1

    return QAList, question_index, Question, Answer
