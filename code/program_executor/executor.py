import os
import pdb
import sys

from IPython.core import ultratb
from utils import get_mapping, get_action_space, get_ind

class Executor():

    def __init__(self,situations, label_dir='../../annotations/STAR_classes/'):

        self.obj_map, self.verb_map = get_mapping(label_dir)
        self.obj_to_ind, self.ind_to_obj, self.ind_to_rel, self.rel_to_ind = get_ind(label_dir)
        self.action_space = get_action_space()
        self.situations = situations
        self._register_modules()
        
    def _register_modules(self):
        self.modules = {
            'Situations':{'func': self.Situations, 'nargs': 0},
            'Actions': {'func': self.Actions, 'nargs': 1},
            'Objs':{'func': self.Objs, 'nargs': 1},
            'Rels':{'func': self.Rels, 'nargs': 1},

            'Filter_Actions_with_Verb': {'func': self.Filter_Actions_with_Verb, 'nargs': 2},
            'Filter_Actions_with_Obj': {'func': self.Filter_Actions_with_Obj, 'nargs': 2},
            'Filter_After_Actions':{'func': self.Filter_After_Actions, 'nargs': 2},
            'Filter_Before_Actions':{'func': self.Filter_Before_Actions, 'nargs': 2},
            'Filter_Situations_with_Rel':{'func': self.Filter_Situations_with_Rel, 'nargs': 2},
            'Filter_Situations_with_Obj':{'func': self.Filter_Situations_with_Obj, 'nargs': 2},
            'Filter_Objs_by_Verb':{'func': self.Filter_Objs_by_Verb, 'nargs': 2},
            
            'Unique': {'func': self.Unique, 'nargs': 1},
            'Equal': {'func': self.Equal, 'nargs': 2},
            'Union':{'func': self.Union, 'nargs': 2},
            'Belong_to':{'func': self.Belong_to, 'nargs': 2},
            'Except':{'func': self.Except, 'nargs': 2},

            'Query_Objs': {'func': self.Query_Objs, 'nargs': 1},
            'Query_Actions': {'func': self.Query_Actions, 'nargs': 1},
            'Query_Verbs': {'func': self.Query_Verbs, 'nargs': 1},
            'Query_Earliest_Action':{'func': self.Query_Earliest_Action, 'nargs': 1},
            'Query_Latest_Action':{'func': self.Query_Latest_Action, 'nargs': 1},
        }

    def pg_transform(self,raw_pg):
        pg = []
        for p in raw_pg:
            pg.extend(p['value_input'])
            pg.extend([p['function']])
        return pg

    def run(self, raw_pg, debug=False):
        exe_stack = []
        argv = []
        pg = self.pg_transform(raw_pg)
        for m in pg:
            if m not in self.modules:
                exe_stack.append(m)
            else:
                argv = []
                for i in range(self.modules[m]['nargs']):
                    if exe_stack:
                        argv.insert(0, exe_stack.pop())
                    else:
                        return 'error'
                step_output = self.modules[m]['func'](*argv)
                if step_output == 'error':
                    return 'error'
                exe_stack.append(step_output)

                if debug:
                    print('> %s%s' % (m, argv))
                    print(exe_stack)

        if exe_stack[-1] is None:
            return 'error'

        return exe_stack[-1]
    
#-------------INPUT MODULE-----------------
    def Situations(self):
        situations = {}
        for frame in self.situations:
            situations[frame]={}
            situations[frame]['actions'] = self.situations[frame]['actions']
            situations[frame]['rel_labels'] = self.situations[frame]['rel_labels']
            situations[frame]['bbox_labels'] = self.situations[frame]['bbox_labels']
            situations[frame]['rel_pairs'] = self.situations[frame]['rel_pairs']
        return situations

#-------------ELEMENT MODULE--------------
    def Actions(self,situations):
        if situations is None:
            return None
        actions = []
        sort_frame = sorted(situations.keys())
        for frame in sort_frame:
            within_actions = situations[frame]['actions']
            for act in within_actions:
                if act not in actions:
                    actions.append(act)
        if actions == []:
            return None
        return actions

    def Objs(self,situations):
        if situations is None:
            return None
        objs = []
        sort_frame = sorted(situations.keys())
        for frame in sort_frame:
            exist_objs = situations[frame]['bbox_labels']
            objs.extend(exist_objs)
            objs = list(set(objs))
        if objs == []:
            return None
        return objs

    def Rels(self,situations):
        if situations is None:
            return None
        rels = []
        sort_frame = sorted(situations.keys())
        for frame in sort_frame:
            exist_rels = situations[frame]['rel_labels']
            rels.extend(exist_rels)
            rels = list(set(rels))
        if rels == []:
            return None
        return rels
#-------------FILTER MODULE--------------
    def Filter_Actions_with_Verb(self,actions,verb):
        if actions is None:
            return None
        filtered_actions = []
        for action in actions:
            if self.verb_map[action]==verb:
                filtered_actions.append(action)
        if filtered_actions == []:
            return None

        return filtered_actions

    def Filter_Actions_with_Obj(self,actions,obj):
        if actions is None:
            return None
        filtered_actions = []
        for action in actions:
            if self.obj_map[action]==obj:
                filtered_actions.append(action)
        if filtered_actions == []:
            return None
        return filtered_actions

    def Filter_After_Actions(self,actions,current_action):

        if actions is None or current_action is None:
            return None
        index = actions.index(current_action)
        if index == len(actions)-1:
            return None
        after_actions= actions[index+1:]

        return after_actions

    def Filter_Before_Actions(self,actions,current_action):
        if actions is None or current_action is None:
            return None
        index = actions.index(current_action)
        if index == 0:
            return None
        before_actions= actions[:index]
        return before_actions

    def Filter_Situations_with_Rel(self,situations,rel):
        if situations is None:
            return None
        select_situation = {}
        for frame in situations:
            relationships = situations[frame]['rel_labels']
            rel_cls = self.rel_to_ind[rel]
            if rel_cls in relationships:
                select_situation[frame] = situations[frame]
        if select_situation == {}:
            return situations
        return select_situation

    def Filter_Situations_with_Obj(self,situations,obj):
        if situations is None:
            return None
        select_situation = {}
        for frame in situations:
            objs = situations[frame]['bbox_labels']
            obj_cls = self.obj_to_ind[obj]
            if obj_cls in objs:
                select_situation[frame] = situations[frame]
        if select_situation == {}:
            return situations

        return select_situation

    def Filter_Objs_by_Verb(self,objs,verb):
        if objs is None:
            return None
        feasible_objs = self.action_space['verb2obj'][verb]
        filtered_obj = []
        for obj_ind in objs:
            if obj_ind not in self.ind_to_obj:
                continue
            obj = self.ind_to_obj[obj_ind]
            if obj in feasible_objs:
                filtered_obj.append(obj)
        if filtered_obj==[]:
            return None

        return filtered_obj
        
#-------------OUTPUT MODULE--------------
    def Query_Verbs(self,action):
        if action is None:
            return None
        if isinstance(action,list):
            verbs = [self.verb_map[act] for act in action]
            return verbs
        return self.verb_map[action]

    def Query_Objs(self,action):
        if action is None:
            return None
        if isinstance(action,list):
            objs = [self.obj_map[act] for act in action]
            return objs
        return self.obj_map[action]

    def Query_Actions(self,action):
        if action is None:
            return [None,None]
        if isinstance(action,list):
            acts = [[self.verb_map[act], self.obj_map[act]] for act in action]
            return acts
        return [self.verb_map[action], self.obj_map[action]]

    def Query_Earliest_Action(self,actions):
        if actions is None:
            return None
        return actions[0]

    def Query_Latest_Action(self,actions):
        if actions is None:
            return None
        return actions[-1]
#-------------LOGICAL MODULE--------------
    def Unique(self,actions):
        if actions is None:
            return None
        return actions[0]

    def Equal(self,item1,item2):
        if item1 is None or item2 is None:
            return 'Wrong'

        if item1 == item2:
            return 'Correct'
        else:
            return 'Wrong'

    def Belong_to(self,group,item):
        if group is None or item is None:
            return 'Wrong'

        if item in group:
            return 'Correct'
        else:
            return 'Wrong'

    def Union(self,item1,item2):
        if item1 is None and item2 is None:
            return None
        if isinstance(item1,list) and isinstance(item2,list):
            return item1+item2
        if isinstance(item1,list) and item2 is None:
            return item1
        if isinstance(item2,list) and item1 is None:
            return item2
        return [item1,item2]

    def Except(self,group1,group2):
        if group1 is None:
            return None
        elif group2 is None:
            return group1
        else:
            except_group=[]
            for item in group1:
                if item in group2:
                    continue
                except_group.append(item)
            if except_group == []:
                return None

            return except_group





