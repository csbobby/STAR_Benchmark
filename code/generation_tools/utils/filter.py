

def filter_action(action_list, data):
    videos = []
    for each in data.keys():
        action_not_found = False
        for each_target_action in action_list:
            found = False
            # print(data[each])
            for each_action in data[each][0]:
                if each_target_action == each_action[0]:
                    found = True
                    break
            if not found:
                action_not_found = True
                break
        if not action_not_found:
            videos.append(each)
    return videos

def filter_verb(verb_list, verb_map, data):
    videos = []
    for each in data.keys():
        action_not_found = False
        for each_target_verb in verb_list:
            found = False
            for each_action in data[each][0]:
                if each_target_verb == verb_map[each_action[0]]:
                    found = True
                    break
            if not found:
                action_not_found = True
                break
        if not action_not_found:
            videos.append(each)
    return videos

def filter_relationship(rel_list, data):
    videos = []
    for each in data.keys():
        relationship_list = []
        if not anno_video_split.__contains__(each):
            continue
        annotation = anno_video_split[each]
        for frame in annotation.keys():
            for i in range(len(annotation[frame])):
                if annotation[frame][i]['attention_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['attention_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
                if annotation[frame][i]['spatial_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['spatial_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
                if annotation[frame][i]['contacting_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['contacting_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
        
        rel_not_found = False
        for each_target_rel in rel_list:
            found = False
            for each_rel in relationship_list:
                if each_target_rel == each_rel:
                    found = True
                    break
            if not found:
                rel_not_found = True
                break
        if not rel_not_found:
            videos.append(each)
    return videos

def filter_all(action_list, verb_list, rel_list, verb_map, data):
    videos = []
    for each in data.keys():
        action_not_found = False
        for each_target_action in action_list:
            found = False
            # print(data[each])
            for each_action in data[each][0]:
                if each_target_action == each_action[0]:
                    found = True
                    break
            if not found:
                action_not_found = True
                break
        if not action_not_found:
            videos.append(each)
            
    videos1 = []
    for each in videos:
        action_not_found = False
        for each_target_verb in verb_list:
            found = False
            for each_action in data[each][0]:
                if each_target_verb == verb_map[each_action[0]]:
                    found = True
                    break
            if not found:
                action_not_found = True
                break
        if not action_not_found:
            videos1.append(each)
            
    videos2 = []
    for each in videos1:
        relationship_list = []
        if not anno_video_split.__contains__(each):
            continue
        annotation = anno_video_split[each]
        for frame in annotation.keys():
            for i in range(len(annotation[frame])):
                if annotation[frame][i]['attention_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['attention_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
                if annotation[frame][i]['spatial_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['spatial_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
                if annotation[frame][i]['contacting_relationship'] is not None:
                    for each_relationship in annotation[frame][i]['contacting_relationship']:
                        relationship_list.append(each_relationship + "," + annotation[frame][i]['class'])
        
        rel_not_found = False
        for each_target_rel in rel_list:
            found = False
            for each_rel in relationship_list:
                if each_target_rel == each_rel:
                    found = True
                    break
            if not found:
                rel_not_found = True
                break
        if not rel_not_found:
            videos2.append(each)
    return videos2


