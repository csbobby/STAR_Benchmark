import copy

# load fps
def get_fps(label_dir):
  return pickle.load(open(label_dir+"/fps", 'rb'))

# loda label mapping
def load_obj_mapping(label_dir,reverse=False):
    map = {}
    with open(label_dir + "/action_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[2]
    obj_map = {}
    with open(label_dir + "/object_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            obj_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = obj_map[map[k]]
    return map

def load_verb_mapping(label_dir,reverse=False):
    map = {}
    with open(label_dir + "/action_mapping.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            map[mapping[0]] = mapping[1]
    verb_map = {}
    with open(label_dir + "/verb_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.split()
            verb_map[mapping[0]] = mapping[1]
    for k in map.keys():
        map[k] = verb_map[map[k]]
    return map

def get_mapping(label_dir):

  obj_map = load_obj_mapping(label_dir)
  verb_map = load_verb_mapping(label_dir)

  return obj_map, verb_map

def get_action_space():

  action_space={'verb2obj': 
  {'wash': ['table','dish','mirror','window','cup/glass/bottle','clothes'],
    'sit': ['table', 'floor', 'bed', 'sofa/couch'],
    'lie': ['bed', 'floor', 'sofa/couch'],
    'tidy': ['blanket', 'clothes', 'towel', 'table', 'closet/cabinet', 'broom'],
    'hold': ['pillow','dish','phone/camera','book','sandwich','food','clothes','shoe','blanket',
     'laptop','mirror','box','broom','picture','medicine'],
    'throw': ['pillow','blanket','clothes','towel','bag','shoe','box','book','food','broom'],
    'put': ['pillow','paper/notebook','book','sandwich','food','dish','clothes','towel','shoe',
     'phone/camera','blanket','laptop','cup/glass/bottle','bag','broom','box','picture'],
    'close': ['book','closet/cabinet','door','refrigerator','window','laptop','box'],
    'take': ['pillow','book','paper/notebook','sandwich','food','towel','clothes','shoe',
     'cup/glass/bottle','phone/camera','dish','blanket','laptop','box','bag','picture','broom'],
    'open': ['book','door','closet/cabinet','refrigerator','bag','laptop','box','window'],
    'eat': ['sandwich', 'medicine']},
   'obj2verb': 
   {'phone/camera': ['hold', 'put', 'take'],
    'picture': ['hold', 'take', 'put'],
    'floor': ['sit', 'lie'],
    'medicine': ['hold', 'eat'],
    'closet/cabinet': ['open', 'close', 'tidy'],
    'book': ['take', 'hold', 'open', 'close', 'put', 'throw'],
    'sandwich': ['eat', 'take', 'hold', 'put'],
    'towel': ['take', 'tidy', 'put', 'throw'],
    'bag': ['take', 'put', 'open', 'throw'],
    'blanket': ['tidy', 'throw', 'put', 'take', 'hold'],
    'mirror': ['hold', 'wash'],
    'laptop': ['take', 'hold', 'put', 'open', 'close'],
    'refrigerator': ['open', 'close'],
    'door': ['open', 'close'],
    'cup/glass/bottle': ['take', 'put', 'wash'],
    'broom': ['hold', 'tidy', 'put', 'take', 'throw'],
    'paper/notebook': ['take', 'put'],
    'dish': ['hold', 'put', 'take', 'wash'],
    'box': ['take', 'hold', 'put', 'open', 'close', 'throw'],
    'pillow': ['take', 'hold', 'put', 'throw'],
    'window': ['close', 'wash', 'open'],
    'food': ['take', 'hold', 'put', 'throw'],
    'bed': ['lie', 'sit'],
    'shoe': ['take', 'put', 'hold', 'throw'],
    'sofa/couch': ['sit', 'lie'],
    'table': ['sit', 'tidy', 'wash'],
    'clothes': ['take', 'hold', 'tidy', 'put', 'throw', 'wash']}}

  return action_space

def get_ind(label_dir):
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

    for i in range(len(obj_vocab)):
        obj_to_ind[obj_vocab[i]] = i
        ind_to_obj[i] = obj_vocab[i]
      
    for i in range(len(rel_vocab)):
        ind_to_rel[i] = rel_vocab[i]
        rel_to_ind[rel_vocab[i]] = i

    return obj_to_ind, ind_to_obj, ind_to_rel, rel_to_ind

