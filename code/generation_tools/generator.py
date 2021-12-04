# -----------------------------------------------------
# STAR Genetarion Tools
# -----------------------------------------------------
import pandas as pd
import numpy as np
import random
import copy
import json

from generation_tools.utils.nlp import * 

RANDOM_SEED = 626
np.random.seed(RANDOM_SEED)
MAX_TIME = 10000
MIN_TIME = -1

class QAGenerator:
    def __init__(self, video_data, anno_video_split, act_map, obj_map,verb_map, action_space, fps, extend_scenegraph, scenegraph, train_answer_space=None ,train_answer_frequence=None,annotation_dir='../annotations/generation_annotations/',avoid=['not_contacting','other_relationship']):
        super(QAGenerator, self).__init__()

        self.video_data = video_data
        self.act_map = act_map
        self.obj_map = obj_map
        self.verb_map = verb_map
        self.actions_with_something_id = ['a009','a038','a044','a081','a082','a096','a108','a126','a127','a102']
        self.min_video_len = 3
        
        self.fps = fps
        self.action_space = action_space
        self.verb_vocab = list(set(action_space['verb']))
        self.obj_vocab = list(set(action_space['object']))
        self.avoid = avoid
        self.min_keyframes = 4

        self.ag_graph = scenegraph
        self.extend_graph = extend_scenegraph
        self.qa_template_dict, self.program_template_dict = self.load_templates(annotation_dir)

        self.anno_video_split = self.generate_initial_annotation(anno_video_split,avoid)
        self.composition_dict = self.generate_action_composition_dict(action_space)
        self.verb_expand, self.verb_perp = self.generate_verb_expand_mapping(annotation_dir)
        self.obj_to_ind, self.ind_to_obj, self.rel_to_ind, self.ind_to_rel = self.generate_graph_cls_mapping(annotation_dir)
        self.realation_inverse = {'in_front_of':'behind','behind':'in_front_of','above':'under','under':'above','beneath':'on','on':'beneath'}

        self.train_answer_frequence = train_answer_frequence
        self.train_answer_space = train_answer_space

#-----------------------INITIALIZE MODULE-------------------------------
    # reorganize the scene graph annotations in ActionGenome to unify the data input/output 
    def generate_initial_annotation(self,annotation,avoid):

        def load_annotation(anno,avoid,rel_type):
            if anno[rel_type] is None:
                return []
            rel_list = list()
            for each_rel in anno[rel_type]:
                if each_rel in avoid:
                    continue
                rel_list.append((anno['class'] + "," + each_rel))
            return rel_list

        initial_anno = {}

        for video in annotation:
            if not self.video_data.__contains__(video):
                continue
            if not initial_anno.__contains__(video):
                initial_anno[video] = {}

            actions = self.video_data[video][0]

            for frame in annotation[video]:
                relationship_dict = {
                'spatial':[],
                'contacting':[],
                'attention':[],
                'verb':[]}
                for i in range(len(annotation[video][frame])):
                    spatial_rels = load_annotation(annotation[video][frame][i],self.avoid,'spatial_relationship')
                    contact_rels = load_annotation(annotation[video][frame][i],self.avoid,'contacting_relationship')
                    attention_rels = load_annotation(annotation[video][frame][i],self.avoid,'attention_relationship')
                    relationship_dict['spatial'].extend(spatial_rels)
                    relationship_dict['contacting'].extend(contact_rels)
                    relationship_dict['attention'].extend(attention_rels)

                for action in actions:
                    action_id,start,end = action[0],action[1],action[2]
                    verb, obj = self.verb_map[action_id], self.obj_map[action_id]
                    if (self.query_time(frame,video) <= float(end)) and (self.query_time(frame,video) >= float(start)):
                        relationship_dict['verb'].append((obj + "," + verb))

                initial_anno[video][frame] = relationship_dict

        return initial_anno

    # build quick-searching dictionary for compositional match
    def generate_action_composition_dict(self,action_space):
        mapping = {'verb2obj':{},'obj2verb':{}}
        verb_vocab = list(set(action_space['verb']))
        obj_vocab = list(set(action_space['object']))
        for verb in verb_vocab:
            index = action_space['verb']==verb
            comp_objs = action_space[index]['object'].unique().tolist()
            mapping['verb2obj'][verb]=comp_objs
        for obj in obj_vocab:
            index = action_space['object']==obj
            comp_verbs = action_space[index]['verb'].unique().tolist()     
            mapping['obj2verb'][obj]=comp_verbs
        return mapping

    def load_templates(self,csv_dir):
        qa_templates = pd.read_csv(csv_dir+'QA_templates.csv',header=None)
        qa_template_dict = {}
        for i in range(len(qa_templates)):
            qtype,temp,qtemp,atemp=qa_templates.iloc[i][0],qa_templates.iloc[i][1],qa_templates.iloc[i][2],qa_templates.iloc[i][3]
            if qtype not in qa_template_dict:
                qa_template_dict[qtype]={}
            qa_template_dict[qtype][temp]={}
            qa_template_dict[qtype][temp]['question'],qa_template_dict[qtype][temp]['answer'] = qtemp,atemp

        program_templates = pd.read_csv(csv_dir+'QA_programs.csv',header=None)
        program_template_dict = {}
        for i in range(len(program_templates)):
            qtype,temp,qtemp,atemp=program_templates.iloc[i][0],program_templates.iloc[i][1],program_templates.iloc[i][2],program_templates.iloc[i][3]
            if qtype not in program_template_dict:
                program_template_dict[qtype]={}
            program_template_dict[qtype][temp]={}
            program_template_dict[qtype][temp]['question'],program_template_dict[qtype][temp]['answer'] = qtemp,atemp

        return qa_template_dict, program_template_dict

    def generate_situation_graphs(self,actions,start_time,end_time,video_id,g_type=None):

        situations = {}
        if g_type == 'extend':
            scene_graph = self.extend_graph
        else:
            scene_graph = self.ag_graph

        if video_id not in scene_graph:
            return None

        select_frames = [frame for frame in list(scene_graph[video_id].keys()) if self.query_time(frame,video_id) >= start_time and self.query_time(frame,video_id) <= end_time ]
        if select_frames == []:
            return None
        actions = sorted(actions,key=lambda x:x[1])
        for frame in select_frames:
            current_time = self.query_time(frame,video_id)
            situations[frame]= copy.deepcopy(scene_graph[video_id][frame])
            situations[frame]['actions'] = [action[0] for action in actions if current_time<=float(action[2]) and current_time>=float(action[1])]
        return situations

    def generate_verb_expand_mapping(self,txt_dir):

        verb_expand = {}
        # for random answers
        verb_perp = {}
        with open(txt_dir + "/verb_expansion.txt") as f:
            lines = f.readlines()
            for line in lines:
                mapping = line.split()
                verb_expand[mapping[0]] = mapping[1] + ' ' + mapping[2]

        with open(txt_dir + "/verb_preposition.txt") as f:
            lines = f.readlines()
            for line in lines:
                mapping = line.split()
                verb_perp[mapping[0]] = mapping[1]

        return verb_expand, verb_perp

    def generate_graph_cls_mapping(self,txt_dir):
        vocabs = []

        with open(txt_dir + "/graph_vocab.txt") as f:
            lines = f.readlines()
            for line in lines:
                vocab = line.strip('\n').split(',')
                vocabs.append(vocab)
        obj_vocab, rel_vocab = vocabs[0], vocabs[1]
        obj_to_ind = {}
        ind_to_obj = {}
        rel_to_ind = {}
        ind_to_rel = {}
        for i in range(len(obj_vocab)):
            obj_to_ind[obj_vocab[i]] = i
            ind_to_obj[i] = obj_vocab[i]
        for i in range(len(rel_vocab)):
            rel_to_ind[rel_vocab[i]] = i
            ind_to_rel[i] = rel_vocab[i]
        return obj_to_ind, ind_to_obj, rel_to_ind, ind_to_rel

    def pass_situation_graph_checking(self,situations,target_action,
        min_action_num=0,action_num_check=True, 
        action_check=True,
        min_frame_num=4,frame_num_check=True):

        if situations is None:
            return False
        if frame_num_check and len(situations.keys())<min_frame_num:
            return False
        extracted_actions = self.extract_actions_from_situiation_graphs(situations)
        if action_num_check and len(extracted_actions)<=min_action_num:
            return False
        if action_check and target_action not in extracted_actions:
            return False
        return True

    def extract_actions_from_situiation_graphs(self,situations):
        extracted = []
        for frame in situations:
            extracted.extend(situations[frame]['actions'])
        return list(set(extracted))
#------------------------------------------------------------------

#-----------------------INPUT MODULE-------------------------------
    def extend_scenegraphs(self,video_id): #MergedGraph
        return self.extend_graph[video_id]

    def scenegraphs(self,video_id): #SceneGraph
        return self.anno_video_split[video_id]

    def actions(self,video_id):
        return self.video_data[video_id][0]
#-------------------------------------------------------------------

#-----------------------FILTER MODULE-------------------------------
    
    #-----------------------FILTER Action-------------------------------
    def filter_actions_by_time(self, actions, mode='after', start=MIN_TIME, end=MAX_TIME): #Filter_actions_by_time
        if actions is None:
            return None 
        if mode=='after':
            filtered_actions = [action for action in actions if (float(action[1]) > start) and (float(action[2]) < end)]
        if mode=='before':
            filtered_actions = [action for action in actions if (float(action[1]) < start) and (float(action[2]) < end)]
        if mode=='in': #within
            filtered_actions = [action for action in actions if (float(action[1]) >= start) and (float(action[2]) <= end)]
        if mode=='cover': # in
            filtered_actions = [action for action in actions if (float(action[1]) <= start) and (float(action[2]) >= end)]
        if mode=='overlap':
            filtered_actions = [action for action in actions if (float(action[1]) <= end) and (float(action[2]) >= start)]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_actions_with_frame(self,actions,frame,video_id):
        current_time = self.query_time(frame,video_id)
        filtered_actions = [action for action in actions if current_time<float(action[2]) and current_time>float(action[1])]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_actions_with_obj(self,actions, obj): #Filter_actions_by_obj
        if actions is None:
            return None
        filtered_actions = [action for action in actions if self.query_obj(action)==obj]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_actions_with_verb(self,actions, verb): #Filter_actions_by_verb
        if actions is None:
            return None
        filtered_actions = [action for action in actions if self.query_verb(action)==verb]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_actions_without_obj(self,actions, obj, avoid=[]): # Blind_actions_by_obj
        if actions is None:
            return None
        filtered_actions = [action for action in actions if self.query_obj(action)!=obj and self.query_obj(action) not in avoid]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_actions_without_verb(self,actions, verb,avoid=[]): # Blind_actions_by_verb
        if actions is None:
            return None
        filtered_actions = [action for action in actions if self.query_verb(action)!=verb and self.query_verb(action) not in avoid]
        if filtered_actions == []:
            return None
        return filtered_actions

    def filter_objs_in_actions(self,actions):
        if actions is None:
            return None
        objs = list(set([self.obj_map[action[0]] for action in actions]))
        return objs

    def filter_verbs_in_actions(self,actions): 
        if actions is None:
            return None
        verbs = list(set([self.verb_map[action[0]] for action in actions]))
        return verbs
    #-------------------------------------------------------------------

    #-----------------------FILTER SCENE GRAPH-------------------------------

    def filter_scenegraph_by_reltype(self,graphs,graph_type):  #Filter_graphs_by_type 
        filter_anno = {}
        for frame in graphs:
            if graphs[frame][graph_type] != []:
                filter_anno[frame] = graphs[frame][graph_type]
        if filter_anno == {}:
            return None
        return filter_anno

    def filter_scenegraph_by_time(self,graphs,video_id, start=MIN_TIME, end=MAX_TIME): #Filter_graphs_by_time
        if graphs is None:
            return None
        filter_anno = {}
        start_frame = self.query_frame(start,video_id)
        end_frame = self.query_frame(end,video_id)
        frame_names = [ frame_name  for frame_name in graphs.keys() if (int(frame_name)<end_frame) and (int(frame_name)>start_frame)]
        for frame in frame_names:
            filter_anno[frame] = graphs[frame]
        if filter_anno == {}:
            return None
        return filter_anno

    def filter_scenegraph_without_obj(self,annotation,obj,avoid=[]): #Blind_graph_by_obj
        if annotation is None:
            return None
        if obj is None:
            return annotation
        filter_anno = {}
        for frame in annotation:
            filter_pairs = [obj_rel for obj_rel in annotation[frame] if (obj_rel.split(',')[0] not in avoid) and (obj_rel.split(',')[0]!=obj)]
            if filter_pairs != []:
                filter_anno[frame] = filter_pairs
        if filter_anno == {}:
            return None
        return filter_anno

    def filter_scenegraph_without_rel(self,annotation,rel,avoid=[]): # Blind_graph_by_relationship
        if annotation is None:
            return None
        if rel is None:
            return annotation
        filter_anno = {}
        for frame in annotation:
            filter_pairs = [obj_rel for obj_rel in annotation[frame] if (obj_rel.split(',')[1] not in avoid) and (obj_rel.split(',')[1]!=rel)]
            if filter_pairs != []:
                filter_anno[frame] = filter_pairs
        if filter_anno == {}:
            return None
        return filter_anno

    def filter_changed_relations(self,start_graph,end_graph):
        if start_graph is None:
            return end_graph
        if end_graph is None:
            return None
        start_rels = start_graph['rel_pairs']
        start_labels = start_graph['rel_labels']
        end_rels = end_graph['rel_pairs']
        end_labels = end_graph['rel_labels']

        changed_rels = []
        changed_rel_labels = []
        for i in range(len(end_rels)):
            rel = end_rels[i]
            if rel not in start_rels:
                changed_rels.append(rel)
                changed_rel_labels.append(end_labels[i])

        if changed_rels == []:
            return None
        return {'rel_pairs':changed_rels,'rel_labels':changed_rel_labels}


    def filter_spatial_relations_in_graphs(self,graphs):
        if graphs is None:
            return None
        filter_graphs = {}
        for frame in graphs:
            filter_rels = []
            filter_rel_labels = []
            for i in range(len(graphs[frame]['rel_pairs'])):
                rel = graphs[frame]['rel_pairs'][i]
                if (0 not in rel) or (rel[1] != 0):
                    continue
                label = graphs[frame]['rel_labels'][i]
                filter_rels.append(rel)
                filter_rel_labels.append(label)
            if filter_rels != []:
                if frame not in filter_graphs:
                    filter_graphs[frame]={}
                filter_graphs[frame]['rel_pairs'] = filter_rels
                filter_graphs[frame]['rel_labels'] = filter_rel_labels
                filter_graphs[frame]['bbox_labels'] = graphs[frame]['bbox_labels']
                filter_graphs[frame]['bbox'] = graphs[frame]['bbox']
        if filter_graphs == {}:
            return None
        return filter_graphs

    def filter_objs_in_graphs(self,graphs): #filter_objs_in_extend_graphs
        if graphs is None:
            return None
        exist_obj = []
        for frame in graphs:
            exist_obj.extend([obj for obj in graphs[frame]['bbox_labels'] if obj != 0] )
        if exist_obj == []:
            return None
        return list(set(exist_obj))
    #-------------------------------------------------------------------

#----------------------------QUERY MODULE--------------------------------------      
    def query_verb(self,action): #Query_verb
        if isinstance(action,list):
            if action[0] in self.verb_expand:
                return self.verb_expand[action[0]]
            return  self.verb_map[action[0]]
        if isinstance(action,str):
            return action.split(",")[1]
    
    def query_obj(self,action): #Query_object
        if isinstance(action,list):
            return self.obj_map[action[0]]
        if isinstance(action,str):
            return action.split(",")[0]

    def query_rel(self,action): #Query_relationship
        return action.split(",")[1]
    
    def query_time(self,frame,video_id):#Query_time
        return float(int(frame)/self.fps[video_id+'.mp4'])

    def query_frame(self,time,video_id):#Query_frame
        return int(time*self.fps[video_id+'.mp4'])

    def query_earliest_action(self,actions):#Query_earliest_action
        if actions is None:
            return None
        action_start_time = [float(action[1]) for action in actions]
        earliest_index = action_start_time.index(min(action_start_time))
        return actions[earliest_index]

    def query_latest_action(self,actions):#Query_latest_action
        if actions is None:
            return None
        action_start_time = [float(action[1]) for action in actions]
        earliest_index = action_start_time.index(max(action_start_time))
        return actions[earliest_index]

    def query_last_graph(self,graphs):
        if graphs is None:
            return None
        return graphs[sorted(graphs.keys())[-1]]

    def query_first_graph(self,graphs):
        if graphs is None:
            return None
        return graphs[sorted(graphs.keys())[0]]

#-------------------------------------------------------------------------

#----------------------------LOGICAL MODULE--------------------------------

    def later(self,compare1,compare2):#Later
        return (float(compare1)>float(compare2))

    def earlier(self,compare1,compare2):#Earlier
        return (float(compare1)<float(compare2))

    def unique(self,items):#Unique
        if isinstance(items[0],list):
            items = [tuple(a) for a in items]
        return len(set(items)) == 1

    def except_(self,set1,set2):
        if set1 is None:
            return None
        if set2 is None:
            return set1
            #avoid_action_id = set2#[act[0] for act in set2]
        actions = [act for act in set1 if act[0] not in set2]
        if actions == []:
            return None
        return actions

    def exist_same_start_action(self,actions):
        start = [action[1] for action in actions]
        return len(start)!=len(set(start))

    def exist_same_end_action(self,actions):
        end = [action[2] for action in actions]
        return len(end)!=len(set(end))

    def exist_inner_actions(self,target_action,actions):
        for action in actions:
            if action[1]>target_action[1]:
                return True
        return False
#-------------------------------------------------------------------------

#----------------------------COMPOSITION MODULE--------------------------------

    def uncompositional_verb(self,obj,avoid_verbs): #Compositional_verb
        avoid_verbs = [verb.split(" ")[0] for verb in avoid_verbs]
        comp_verbs = copy.deepcopy(self.composition_dict['obj2verb'][obj])
        vocabs = copy.deepcopy(self.verb_vocab)
        uncomp_verbs = [verb for verb in vocabs if (verb not in avoid_verbs) and (verb not in comp_verbs)]
        uncomp_verb = uncomp_verbs[np.random.randint(0,len(uncomp_verbs))]
        if uncomp_verb in self.verb_perp:
            uncomp_verb = uncomp_verb + self.verb_perp[uncomp_verb]
        return uncomp_verb

    def uncompositional_obj(self,verb,avoid): #Compositional_obj
        verb = verb.split(" ")[0]
        comp_objs = copy.deepcopy(self.composition_dict['verb2obj'][verb])
        vocabs = copy.deepcopy(self.obj_vocab)
        uncomp_objs = [obj for obj in vocabs if (obj not in avoid) and (obj not in comp_objs)]
        uncomp_obj = uncomp_objs[np.random.randint(0,len(uncomp_objs))]
        return uncomp_obj

#-------------------------------------------------------------------------

#----------------------------QAO Generation--------------------------------

    def generate_question(self,qtype,temp,question_words):
        question = copy.deepcopy(self.qa_template_dict[qtype][temp]['question'])
        program = copy.deepcopy(self.program_template_dict[qtype][temp]['question'])
        token_mapping = {'obj':'[Obj]','verb':'[Verb]','verbpresent':'[Verb]ing','verbpast':'[Verb]ed','conrel':'[Contact_Rel]','sparel':'[Spatial_Rel]',
        'obj1':'[Obj1]','verb1':'[Verb1]','verb1present':'[Verb1]ing','verb1past':'[Verb1]ed','conrel1':'[Contact_Rel1]',
        'obj2':'[Obj2]','verb2':'[Verb2]','verb2present':'[Verb2]ing','verb2past':'[Verb2]ed','conrel2':'[Contact_Rel2]',
        'verbparti':'[Verb]ed','verb1parti':'[Verb1]ed','verb2parti':'[Verb2]ed'}

        for key in question_words:
            template_key = token_mapping[key]
            if 'parti' in key and 'ed' in template_key:
                question_verb=question_words[key]
                converted_verb = question_verb.replace(question_verb.split(" ")[0], conjugate(question_verb.split(" ")[0], tense = PAST+PARTICIPLE, alias = '3sg') ) # taken
                question = question.replace(template_key,converted_verb)
                template_key=template_key[:-2]
                program = program.replace(template_key,question_verb.split(" ")[0])

            if 'parti' not in key and 'ed' in template_key:
                question_verb=question_words[key]
                converted_verb = question_verb.replace(question_verb.split(" ")[0], conjugate(question_verb.split(" ")[0], tense = PAST, alias = '3sg') ) # taken
                question = question.replace(template_key,converted_verb)
                template_key=template_key[:-2]
                program = program.replace(template_key,question_verb.split(" ")[0])
            elif 'ing' in template_key:
                question_verb=question_words[key]
                converted_verb = question_verb.replace(question_verb.split(" ")[0], conjugate(question_verb.split(" ")[0], tense = PRESENT, alias = '3sg', aspect = PROGRESSIVE)) # taking
                question = question.replace(template_key,converted_verb)
                template_key=template_key[:-3]
                program = program.replace(template_key,question_verb.split(" ")[0])
            else:
                question = question.replace(template_key,question_words[key])
                program = program.replace(template_key,question_words[key].split(" ")[0])

        question = question.replace("_"," ")

        return question, program.split(',')


    def generate_correct_answer(self,qtype,temp,answer_words,template_type,obj_index='',verb_index=''):
        correct_answer = copy.deepcopy(self.qa_template_dict[qtype][temp]['answer'])
        program = copy.deepcopy(self.program_template_dict[qtype][temp]['answer'])
        template_type = template_type.split('_')
        if 'obj' in template_type:
            target_obj = answer_words['obj']
            template_token = '[Obj'+obj_index+']'
            correct_answer = correct_answer.replace(template_token,target_obj)
            program = program.replace(template_token,target_obj)
            return_word = target_obj
            
        if 'verbpast' in template_type:
            target_verb = answer_words['verb']
            converted_verb = target_verb.replace(target_verb.split(" ")[0], conjugate(target_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            template_token = '[Verb'+verb_index+']ed'
            correct_answer = correct_answer.replace(template_token,converted_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,target_verb.split(" ")[0])
            return_word = target_verb.split(" ")[0]

        if 'verb' in template_type:
            target_verb = answer_words['verb']
            template_token = '[Verb'+verb_index+']'
            correct_answer = correct_answer.replace(template_token,target_verb)
            program = program.replace(template_token,target_verb.split(" ")[0])
            return_word = target_verb.split(" ")[0]

        correct_answer = correct_answer.capitalize()

        return correct_answer, program.split(','), return_word

    def generate_conflict_option(self,
        qtype,temp,
        question_words,answer_words,
        exists_options,
        template_type,obj_index='',verb_index=''):

        conflict_option = copy.deepcopy(self.qa_template_dict[qtype][temp]['answer'])
        program = copy.deepcopy(self.program_template_dict[qtype][temp]['answer'])
        template_type = template_type.split('_')

        if 'fix-type1' in template_type:
            conflict_obj = answer_words['obj']
            target_verb = answer_words['verb']
            if isinstance(target_verb,str):
                target_verb = [target_verb]
            conflict_verb = self.compositional_verb(conflict_obj,target_verb)
            if conflict_verb is None:
                conflict_verb = self.random_verb(target_verb)
            converted_verb = conflict_verb.replace(conflict_verb.split(" ")[0], conjugate(conflict_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            conflict_option = conflict_option.replace('[Verb]ed', converted_verb).replace('[Obj]',conflict_obj)
            conflict_option = conflict_option.capitalize()
            program = program.replace('[Obj]',conflict_obj).replace('[Verb]',conflict_verb.split(" ")[0])
            return_word = conflict_verb.split(" ")[0]

        if 'fix-type2' in template_type:
            conflict_obj = answer_words['obj']
            target_verb = answer_words['verb']
            if isinstance(target_verb,str):
                target_verb = [target_verb]
            conflict_verb = self.compositional_verb(conflict_obj,target_verb)
            if conflict_verb is None:
                conflict_verb = self.random_verb(target_verb)
            converted_verb = conflict_verb.replace(conflict_verb.split(" ")[0], conjugate(conflict_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            conflict_option = conflict_option.replace('[Verb]ed', converted_verb).replace('[Obj]',conflict_obj)
            conflict_option = conflict_option.capitalize()
            if self.out_of_vocab(qtype,temp,conflict_option):
                conflict_option = self.random_answer(qtype,temp,exists_options)
            conflict_verb = conflict_option.lower().strip('.').split(" ")[0]
            conflict_verb = lemma(conflict_verb)
            program = program.replace('[Obj]',conflict_obj).replace('[Verb]',conflict_verb.split(" ")[0]) 
            return_word = conflict_verb.split(" ")[0]

        if 'fix-type3' in template_type:
            conflict_obj = answer_words['obj']
            conflict_verb = answer_words['verb']
            template_token = '[Verb'+verb_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_verb)
            program = program.replace(template_token,conflict_verb.split(" ")[0])
            template_token = '[Obj'+verb_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_obj)
            program = program.replace(template_token,conflict_obj)
            conflict_option = conflict_option.capitalize()
            return_word = None

        if 'fix-type7' in template_type:
            conflict_obj = answer_words['obj']
            conflict_verb = answer_words['verb']
            template_token = '[Verb'+verb_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_verb)
            template_token = '[Obj'+verb_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_obj)
            conflict_option = conflict_option.capitalize()
            if self.out_of_vocab(qtype,temp,conflict_option):
                conflict_option = self.random_answer(qtype,temp,exists_options)
            template_token = '[Verb'+verb_index+']'
            conflict_verb = conflict_option.lower().strip('.').split(" ")[0]
            conflict_verb = lemma(conflict_verb)
            program = program.replace(template_token,conflict_verb.split(" ")[0])
            template_token = '[Obj'+verb_index+']'
            conflict_obj = conflict_option.lower().strip('.').split(" ")[-1]
            program = program.replace(template_token,conflict_obj)
            return_word = None

        if 'fix-type4' in template_type:
            conflict_verb = answer_words['verb']
            template_token = '[Verb'+verb_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_verb)
            conflict_option = conflict_option.capitalize()
            program = program.replace(template_token,conflict_verb.split(" ")[0])
            return_word = conflict_verb.split(" ")[0]

        if 'fix-type5' in template_type:
            conflict_obj = answer_words['obj']
            template_token = '[Obj'+obj_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_obj)
            conflict_option = conflict_option.capitalize()
            program = program.replace(template_token,conflict_obj)
            return_word = conflict_obj

        if 'fix-type6' in template_type:
            conflict_obj = answer_words['obj']
            template_token = '[Obj'+obj_index+']'
            conflict_option = conflict_option.replace(template_token,conflict_obj)
            conflict_option = conflict_option.capitalize()
            if self.out_of_vocab(qtype,temp,conflict_option):
                conflict_option = self.random_answer(qtype,temp,exists_options)
            conflict_obj = conflict_option.lower().strip('.').split(" ")[-1]
            template_token = '[Obj'+obj_index+']'
            program = program.replace(template_token,conflict_obj)
            return_word = conflict_obj

        if 'obj' in template_type:
            question_verb = question_words['verb']
            target_obj = answer_words['obj']
            if isinstance(target_obj,str):
                target_obj = [target_obj]
            template_token = '[Obj'+obj_index+']'
            conflict_obj = self.compositional_obj(question_verb,target_obj)
            if conflict_obj is None:
                conflict_obj = self.random_obj(target_obj)
            conflict_option = conflict_option.replace(template_token,conflict_obj)
            conflict_option = conflict_option.capitalize()
            if self.out_of_vocab(qtype,temp,conflict_option):
                conflict_option = self.random_answer(qtype,temp,exists_options)
                conflict_obj = conflict_option.lower().strip('.').split(' ')[-1]
            program = program.replace(template_token,conflict_obj)
            return_word = conflict_obj

        if 'verb' in template_type:
            question_obj = question_words['obj']
            target_verb = answer_words['verb']
            if isinstance(target_verb,str):
                target_verb = [target_verb]
            template_token = '[Verb'+verb_index+']'
            conflict_verb = self.compositional_verb(question_obj,target_verb)
            if conflict_verb is None:
                conflict_verb = self.random_verb(target_verb)
            conflict_option = conflict_option.replace(template_token,conflict_verb)
            conflict_option = conflict_option.capitalize()
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,conflict_verb.split(" ")[0])
            return_word = conflict_verb.split(" ")[0]
           
        if 'verbpast' in template_type:
            question_obj = question_words['obj']
            target_verb = answer_words['verb']
            if isinstance(target_verb,str):
                target_verb = [target_verb]
            template_token = '[Verb'+verb_index+']ed'
            conflict_verb = self.compositional_verb(question_obj,target_verb)
            if conflict_verb is None:
                conflict_verb = self.random_verb(target_verb)
            converted_verb = conflict_verb.replace(conflict_verb.split(" ")[0], conjugate(conflict_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            conflict_option = conflict_option.replace(template_token,converted_verb)
            conflict_option = conflict_option.capitalize()
            if self.out_of_vocab(qtype,temp,conflict_option):
                conflict_option = self.random_answer(qtype,temp,exists_options)
            conflict_verb = conflict_option.lower().strip('.').split(" ")[0]
            conflict_verb = lemma(conflict_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,conflict_verb.split(" ")[0])
            return_word = conflict_verb.split(" ")[0]

        if 'action' in template_type:
            conflict_option = self.random_answer(qtype,temp,exists_options)
            conflict_verb = conflict_option.lower().strip('.').split(" ")[0]
            conflict_verb = lemma(conflict_verb)
            conflict_obj = conflict_option.lower().strip('.').split(" ")[-1]
            template_token = '[Obj'+obj_index+']'
            program = program.replace(template_token,conflict_obj)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,conflict_verb)
            return_word = None

        return conflict_option, program.split(','), return_word

    def generate_frequent_option(self,
        qtype,temp,
        q_key,
        correct_option_key,conflict_option_key,
        question_words,
        exist_words,
        exist_options,
        template_type,obj_index='',verb_index=''):

        fre_option = copy.deepcopy(self.qa_template_dict[qtype][temp]['answer'])
        program = copy.deepcopy(self.program_template_dict[qtype][temp]['answer'])
        template_type = template_type.split('_')

        if 'fix-type1' in template_type:
            obj = question_words['obj']
            fre_verb = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if fre_verb is None:
                fre_verb = self.compositional_verb(obj,exist_words)
                if fre_verb is None:
                    fre_verb = self.random_verb(exist_words)
            converted_verb = fre_verb.replace(fre_verb.split(" ")[0], conjugate(fre_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            fre_option = fre_option.replace('[Verb]ed',converted_verb).replace('[Obj]',obj)
            fre_option = fre_option.capitalize()
            program = program.replace('[Obj]',obj).replace('[Verb]',fre_verb.split(" ")[0])
            return_word = fre_verb.split(" ")[0]

        if 'fix-type2' in template_type:
            obj = question_words['obj'] #answer_words
            fre_action = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if fre_action is None:
                fre_verb = self.compositional_verb(obj,exist_words)
                if fre_verb is None:
                    fre_verb = self.random_verb(exist_words)
                converted_verb = fre_verb.replace(fre_verb.split(" ")[0], conjugate(fre_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
                fre_option = fre_option.replace('[Verb]ed',converted_verb).replace('[Obj]',obj)
                fre_option = fre_option.capitalize()
            else:
                fre_option = fre_action
            if self.out_of_vocab(qtype,temp,fre_option):
                fre_option = self.random_answer(qtype,temp,exist_options)
            fre_verb = fre_option.lower().split(" ")[0]
            fre_verb = lemma(fre_verb)
            fre_obj = fre_option.lower().strip('.').split(" ")[-1]
            program = program.replace('[Obj]',fre_obj).replace('[Verb]',fre_verb.split(" ")[0])
            return_word = fre_verb.split(" ")[0]

        if 'fix-type3' in template_type:
            fre_obj = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            template_token = '[Obj'+obj_index+']'
            if fre_obj is None:
                fre_option = self.random_answer(qtype,temp,exist_options)
            else:
                fre_option = fre_option.replace(template_token, fre_obj)
                fre_option = fre_option.capitalize()
            fre_obj = fre_option.lower().strip('.').split(' ')[-1]
            program = program.replace(template_token,fre_obj)
            return_word = fre_obj

        if 'obj' in template_type:
            fre_obj = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if obj_index=='1':
                question_obj=exist_words[1]
            if obj_index=='2':
                question_obj=exist_words[0]
            else:
                question_obj=''
            if fre_obj is None or fre_obj==question_obj:
                fre_obj = self.compositional_obj(question_words['verb'],exist_words)
                if fre_obj is None:
                    fre_obj = self.random_obj(exist_words)
            template_token = '[Obj'+obj_index+']'
            fre_option = fre_option.replace(template_token, fre_obj)
            fre_option = fre_option.capitalize()
            if self.out_of_vocab(qtype,temp,fre_option):
                fre_option = self.random_answer(qtype,temp,exist_options)
            fre_obj = fre_option.lower().strip('.').split(' ')[-1]
            program = program.replace(template_token,fre_obj)
            return_word = fre_obj

        if 'verbpast' in template_type:
            fre_verb = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if verb_index=='1':
                question_verb=exist_words[1]
            if verb_index=='2':
                question_verb=exist_words[0]
            else:
                question_verb=''
            if fre_verb is None or fre_verb==question_verb:
                fre_verb = self.compositional_verb(question_words['obj'],exist_words)
                if fre_verb is None:
                    fre_verb = self.random_verb(exist_words)
            converted_verb = fre_verb.replace(fre_verb.split(" ")[0], conjugate(fre_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            template_token = '[Verb'+verb_index+']ed'
            fre_option = fre_option.replace(template_token,converted_verb)
            fre_option = fre_option.capitalize()
            if self.out_of_vocab(qtype,temp,fre_option):
                fre_option = self.random_answer(qtype,temp,exist_options)
            fre_verb = fre_option.lower().strip('.').split(" ")[0]
            fre_verb = lemma(fre_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,fre_verb.split(" ")[0])
            return_word = fre_verb.split(" ")[0]

        if 'uncompverb' in template_type:
            fre_obj = question_words['obj']
            fre_verb = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if fre_verb is None:
                fre_verb = self.uncompositional_verb(fre_obj,exist_words)
            template_token = '[Obj'+obj_index+']'
            fre_option = fre_option.replace(template_token, fre_obj)
            program = program.replace(template_token,fre_obj)
            template_token = '[Verb'+verb_index+']'
            fre_option = fre_option.replace(template_token, fre_verb)
            program = program.replace(template_token,fre_verb.split(' ')[0])
            fre_option = fre_option.capitalize()
            return_word = fre_verb.split(' ')[0]

        if 'verb' in template_type:
            fre_verb = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if verb_index=='1':
                question_verb=exist_words[1]
            if verb_index=='2':
                question_verb=exist_words[0]
            else:
                question_verb=''
            if fre_verb is None or fre_verb==question_verb:
                fre_verb = self.compositional_verb(question_words['obj'],exist_words)
                if fre_verb is None:
                    fre_verb = self.random_verb(exist_words)
            template_token = '[Verb'+verb_index+']'
            fre_option = fre_option.replace(template_token,fre_verb)
            fre_option = fre_option.capitalize()
            if self.out_of_vocab(qtype,temp,fre_option):
                fre_option = self.random_answer(qtype,temp,exist_options)
            fre_verb = fre_option.lower().strip('.').split(" ")[0]
            fre_verb = lemma(fre_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,fre_verb.split(" ")[0])
            return_word = fre_verb.split(" ")[0]

        if 'action' in template_type:
            fre_option = self.get_frequent_answer(qtype,temp,q_key,correct_option_key,conflict_option_key)
            if fre_option is None:
                fre_option = self.random_answer(qtype,temp,exist_options)
                fre_option = fre_option.capitalize()
            fre_verb = fre_option.lower().strip('.').split(" ")[0]
            fre_verb = lemma(fre_verb)
            fre_obj = fre_option.lower().strip('.').split(' ')[-1]
            template_token = '[Verb'+obj_index+']'
            program = program.replace(template_token,fre_verb.split(" ")[0])
            template_token = '[Obj'+obj_index+']'
            program = program.replace(template_token,fre_obj)
            return_word = None


        return fre_option, program.split(','), return_word

    def generate_random_option(self,
        qtype,temp,
        question_words,
        exist_words,exist_options,
        template_type,obj_index='',verb_index=''):

        random_option = self.qa_template_dict[qtype][temp]['answer']
        program = self.program_template_dict[qtype][temp]['answer']
        template_type = template_type.split('_')

        if 'fix-type1' in template_type:
            obj = question_words['obj']
            random_verb = self.compositional_verb(obj,exist_words)
            if random_verb is None:
                random_verb = self.random_verb(exist_words)
            converted_verb = random_verb.replace(random_verb.split(" ")[0], conjugate(random_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            random_option = random_option.replace('[Verb]ed', converted_verb).replace('[Obj]',obj)
            random_option = random_option.capitalize()
            program = program.replace('[Obj]',obj).replace('[Verb]',random_verb.split(" ")[0])

        if 'fix-type2' in template_type:
            obj = question_words['obj']
            random_verb = self.compositional_verb(obj,exist_words)
            if random_verb is None:
                random_verb = self.random_verb(exist_words)
            converted_verb = random_verb.replace(random_verb.split(" ")[0], conjugate(random_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            random_option = random_option.replace('[Verb]ed', converted_verb).replace('[Obj]',obj)
            random_option = random_option.capitalize()
            if self.out_of_vocab(qtype,temp,random_option):
                random_option = self.random_answer(qtype,temp,exist_options)
            random_verb = random_option.lower().split(" ")[0]
            random_verb = lemma(random_verb)
            random_obj = random_option.lower().strip('.').split(" ")[-1]
            program = program.replace('[Obj]',random_obj).replace('[Verb]',random_verb.split(" ")[0])

        if 'obj' in template_type:
            random_obj = self.compositional_obj(question_words['verb'],exist_words)
            template_token = '[Obj'+obj_index+']'
            if random_obj is None:
                random_obj = self.random_obj(exist_words)
            random_option = random_option.replace(template_token, random_obj)
            random_option = random_option.capitalize()
            if self.out_of_vocab(qtype,temp,random_option):
                random_option = self.random_answer(qtype,temp,exist_options)
            random_obj = random_option.lower().strip('.').split(' ')[-1]
            program = program.replace(template_token,random_obj)

        if 'uncompobj' in template_type:
            random_obj = self.uncompositional_obj(question_words['verb'],exist_words)
            template_token = '[Obj'+obj_index+']'
            random_option = random_option.replace(template_token,random_obj)
            random_option = random_option.capitalize()        
            if self.out_of_vocab(qtype,temp,random_option):
                random_option = self.random_answer(qtype,temp,exist_options)
            random_obj = random_option.lower().strip('.').split(' ')[-1]
            program = program.replace(template_token,random_obj)

        if 'verbpast' in template_type:
            random_verb = self.compositional_verb(question_words['obj'],exist_words)
            if random_verb is None:
                random_verb = self.random_verb(exist_words)
            converted_verb = random_verb.replace(random_verb.split(" ")[0], conjugate(random_verb.split(" ")[0], tense = PAST, alias = '3sg')) # took
            template_token = '[Verb'+verb_index+']ed'
            random_option = random_option.replace(template_token,converted_verb)
            random_option = random_option.capitalize()
            if self.out_of_vocab(qtype,temp,random_option):
                random_option = self.random_answer(qtype,temp,exist_options)
            random_verb = random_option.lower().strip('.').split(" ")[0]
            random_verb = lemma(random_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,random_verb.split(" ")[0])

        if 'verb' in template_type:
            random_verb = self.compositional_verb(question_words['obj'],exist_words)
            if random_verb is None:
                random_verb = self.random_verb(exist_words)
            template_token = '[Verb'+verb_index+']'
            random_option = random_option.replace(template_token,random_verb)
            random_option = random_option.capitalize()
            if self.out_of_vocab(qtype,temp,random_option):
                random_option = self.random_answer(qtype,temp,exist_options)
            random_verb = random_option.lower().strip('.').split(" ")[0]
            random_verb = lemma(random_verb)
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,random_verb.split(" ")[0])

        if 'uncompverb' in template_type:
            random_obj = question_words['obj']
            random_verb = self.uncompositional_verb(random_obj,exist_words)
            template_token = '[Verb'+obj_index+']'
            random_option = random_option.replace(template_token,random_verb)
            program = program.replace(template_token,random_verb.split(' ')[0])
            template_token = '[Obj'+obj_index+']'
            random_option = random_option.replace(template_token,random_obj)
            program = program.replace(template_token,random_obj)
            random_option = random_option.capitalize()        

        if 'action' in template_type:
            random_option = self.random_answer(qtype,temp,exist_options)
            random_verb = random_option.lower().strip('.').split(" ")[0]
            random_verb = lemma(random_verb)
            random_obj = random_option.lower().strip('.').split(' ')[-1]
            template_token = '[Verb'+verb_index+']'
            program = program.replace(template_token,random_verb.split(" ")[0])
            template_token = '[Obj'+obj_index+']'
            program = program.replace(template_token,random_obj)

        return random_option, program.split(',')

    def out_of_vocab(self,qtype,temp,option):
        candiates = copy.deepcopy(self.train_answer_space[qtype][temp])
        return option not in candiates

    def compositional_verb(self,obj,current_verbs):
        current_verbs = [verb.split(" ")[0] for verb in current_verbs]
        if obj not in self.composition_dict['obj2verb']:
            return None
        comp_verbs = copy.deepcopy(self.composition_dict['obj2verb'][obj])
        candiate_verbs = [verb for verb in comp_verbs if verb not in current_verbs]
        if len(candiate_verbs) ==0:
            return None
        comp_verb = candiate_verbs[np.random.randint(0,len(candiate_verbs))]
        #print('candiate_verbs',candiate_verbs)
        #print('comp_verbs',comp_verbs)
        if comp_verb in self.verb_perp:
            comp_verb = comp_verb + ' ' +self.verb_perp[comp_verb]
        return comp_verb

    def compositional_obj(self,verb,current_objs):
        verb = verb.split(" ")[0]
        if verb not in self.composition_dict['verb2obj']:
            return None
        comp_objs = copy.deepcopy(self.composition_dict['verb2obj'][verb])
        candiate_objs = [obj for obj in comp_objs if obj not in current_objs]
        if len(candiate_objs) == 0:
            return None
        comp_obj = candiate_objs[np.random.randint(0,len(candiate_objs))]
        return comp_obj

    def random_obj(self,avoid_objs):
        copyed_vocab = copy.deepcopy(self.obj_vocab)
        candiates = [obj for obj in copyed_vocab if obj not in avoid_objs] 
        random_obj = candiates[np.random.randint(0,len(candiates))]
        return random_obj

    def random_verb(self,avoid_verbs): 
        avoid_verbs = [verb.split(" ")[0] for verb in avoid_verbs] # remove preposition
        copyed_vocab = copy.deepcopy(self.verb_vocab)
        candiates = [verb for verb in copyed_vocab if verb not in avoid_verbs] 
        random_verb = candiates[np.random.randint(0,len(candiates))]
        if random_verb in self.verb_perp:
            random_verb = random_verb + ' ' +self.verb_perp[random_verb]
        return random_verb

    def random_answer(self,qtype,temp,exists_options):
        all_answer = copy.deepcopy(self.train_answer_space[qtype][temp])
        candiates = [ans for ans in all_answer if ans not in exists_options]
        rand_option = candiates[np.random.randint(0,len(candiates))]
        return rand_option

    def get_frequent_answer(self,qtype,temp,qkey,correct,confict):
        if qkey not in self.train_answer_frequence[qtype][temp]:
            return None
        a_key_stat = copy.deepcopy(self.train_answer_frequence[qtype][temp][qkey])
        sorted_a_key = sorted(a_key_stat.items(),key=lambda x:x[1])
        sorted_candiactes = [ items[0] for items in sorted_a_key]
        ans_index = sorted_candiactes.index(correct) if correct in sorted_candiactes else -2
        conf_index = sorted_candiactes.index(confict)  if confict in sorted_candiactes else -2
        #print(ans_index,conf_index)
        max_index = max([ans_index,conf_index])
        min_index = min([ans_index,conf_index])

        if len(sorted_candiactes)==1:
            return None
        elif len(sorted_candiactes)==2 and confict in sorted_candiactes and correct in sorted_candiactes:
            return None
        elif max_index == len(sorted_candiactes)-1 and min_index == len(sorted_candiactes)-2:
            fre_index = np.random.randint(0,min_index)
            fre = sorted_candiactes[fre_index]
        elif max_index == len(sorted_candiactes)-1 and min_index != len(sorted_candiactes)-2:
            if min_index == -2:
                fre_index = np.random.randint(0, max_index)
            else:
                fre_index = np.random.randint(min_index+1, max_index)
            fre = sorted_candiactes[fre_index]
        elif max_index != len(sorted_candiactes)-1:
            fre_index = np.random.randint(max_index+1,len(sorted_candiactes))
            fre = sorted_candiactes[fre_index]
        if fre in self.verb_perp:
            fre = fre + ' ' +self.verb_perp[fre]

        return fre

        
