import random
import numpy as np
from aug_utils import *
from utils.basic_utils import load_jsonl, l2_normalize_np_array, l2_normalize_torch_tensor
import torch


def bg_mix(data, moments, non_moments, ctx_l, clip_len, db_range, moment_db, text_feature):
    '''
    to replace Non GT to other videos' GT
    
    '''

    ###############################################
    # non_moment_segments replacement
    ###############################################

    short_sample = False
    for (s, e) in moments:
        if (e-s) <= 30:
            short_sample = True
            break
    
    if not short_sample:
        return None

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        non_moment = dict()

        need_len = (e- s)

        find = False
        db_range_idx = -1
        for db_range_ in db_range:
            if need_len > db_range_:
                find = True
                break
            db_range_idx += 1

        if not find or db_range_idx == -1:
            # print(need_len)
            return None
        
        while True:
            another_moment = random.choice(moment_db[db_range_idx])
            if another_moment[0] != data['qid'] and another_moment[1] != data['vid']:
                break

        non_moment['vid'] = another_moment[1]


        ass, aee = another_moment[2]
        if aee - ass < need_len:
            assert False
        else:
            aee = ass + need_len

        if ass == 0:
            rss = 0
        else:
            rss = int(ass // clip_len) if ass % clip_len == 0 else int(ass // clip_len) + 1

        ree = int(aee // clip_len)


        non_moment['clip_id'] = [rss, ree]
        non_moment['len'] = (ree - rss)

        non_moment_segments.append(non_moment)


    ###############################################
    # Generate new data dict
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']
    new_data['relevant_windows'] = data['relevant_windows']

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = data['relevant_clip_ids']

    if 'saliency_scores' in data:
        new_data['saliency_scores'] = data['saliency_scores']


    new_data['org_clip_ids_order'] = []

    non_moments_idx = 0
    if non_moments[0][0] == 0:
        non_moment_segment = non_moment_segments[0]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    for i in range(len(moments) - 1):

        # moment segment
        s, e = moments[i]
        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

        # non-moment segment
        non_moment_segment = non_moment_segments[non_moments_idx]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    # moment segment
    s, e = moments[-1]
    rs = int(s // clip_len) if s != 0 else 0
    re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
    new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

    if non_moments_idx < len(non_moment_segments):
        non_moment_segment = non_moment_segments[-1]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))

    assert len(new_data['org_clip_ids_order']) == len(moments) + len(non_moments)
    return new_data
