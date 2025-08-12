import random
import numpy as np
from aug_utils import *

def fg_mix(data, moments, non_moments, thres_crop, ctx_l, clip_len):
    '''
    In a video, crop one longer moment and mix non-moment and moments
    
    '''

    ###############################################
    # Find long moment
    ###############################################

    max_moment_length = 0
    max_moment_idx = -1
    ms, me = -1, -1
    for i, (s, e) in enumerate(moments):
        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        rs, re = rs * clip_len, re * clip_len

        l = re - rs
        if max_moment_length < l:
            max_moment_length = l
            max_moment_idx = i
            ms, me = rs, re

    # If no long moments exist, this data cannot be split
    if max_moment_length < thres_crop * 2:
        return None

    ###############################################
    # Split long moments and create moment segments
    ###############################################

    num_crop = max_moment_length // thres_crop - 1
    moment_crop_idxs = crop_clip_index(ms, me, num_crop=num_crop, clip_len=clip_len)

    moment_segments = []

    ss_idx = 0
    for i, (s, e) in enumerate(moments):
        if i == max_moment_idx:
            moment_crop_idxs.append(e)
            ss = s
            for ee in moment_crop_idxs:
                moment = dict()
                
                rss = int(ss // clip_len) if ss != 0 else 0
                ree = int(ee // clip_len) if ee % clip_len == 0 else int(ee // clip_len) + 1
                if clip_len < 1 and s != ss: # vgg
                    rss += 1
                moment['clip_id'] = [rss, ree]
                moment['seg_sec'] = [ss - rss * clip_len, ree * clip_len - ee]
                moment['len'] = (ree - rss)

                if 'saliency_scores' in data:
                    ss_nxt_idx = ss_idx + moment['len']
                    moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
                    ss_idx = ss_nxt_idx

                moment_segments.append(moment)
                ss = ee
        else:
            moment = dict()

            rs = int(s // clip_len) if s != 0 else 0
            re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
            moment['clip_id'] = [rs, re]
            moment['seg_sec'] = [s - rs * clip_len, re * clip_len - e]
            moment['len'] = (re - rs)

            if 'saliency_scores' in data:
                ss_nxt_idx = ss_idx + moment['len']
                moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
                ss_idx = ss_nxt_idx

            moment_segments.append(moment)


    ###############################################
    # Split long non-moments into required segments
    ###############################################

    need_crop_count = len(moment_segments) + 1 - len(non_moments)

    non_moment_crop_idxs = []
    non_moment_idxs = []

    for i, (s, e) in enumerate(non_moments):

        if s == 0:
            rs = 0
        else:
            rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
        re = int(e // clip_len)
        rs, re = rs * clip_len, re * clip_len

        l = re - rs
        if l >= thres_crop * 2:
            num_crop = min(l // thres_crop - 1, need_crop_count)
            non_moment_crop_idxs.append(crop_clip_index(rs, re, num_crop=num_crop, clip_len=clip_len))
            non_moment_idxs.append(i)

            need_crop_count -= num_crop
        
        if need_crop_count <= 0:
            break
    # If there are not enough non-moments to fill the cropped moments, this data cannot be used
    if need_crop_count > 0 :
        return None


    ###############################################
    # Create non_moment_segments
    ###############################################

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        if i in non_moment_idxs:
            _non_moment_crop_idxs = non_moment_crop_idxs[non_moment_idxs.index(i)]
            _non_moment_crop_idxs.append(e)
            
            ss = s
            for ee in _non_moment_crop_idxs:
                non_moment = dict()
                
                if ss == 0:
                    rss = 0
                else:
                    rss = int(ss // clip_len) if ss % clip_len == 0 else int(ss // clip_len) + 1
                ree = int(ee // clip_len)
                if clip_len < 1 and s != ss: # vgg
                    rss -= 1

                non_moment['clip_id'] = [rss, ree]
                non_moment['len'] = (ree - rss)

                non_moment_segments.append(non_moment)
                ss = ee
        else:
            non_moment = dict()

            if s == 0:
                rs = 0
            else:
                rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
            re = int(e // clip_len)

            non_moment['clip_id'] = [rs, re]
            non_moment['len'] = (re - rs)

            non_moment_segments.append(non_moment)



    ###############################################
    # Mix moments and non-moments
    ###############################################

    random.shuffle(non_moment_segments)
    random.shuffle(moment_segments)

    ###############################################
    # Generate new data dict
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']

    new_clips = np.zeros(ctx_l)

    # new_data['saliency_scores'] ok
    # new_data['org_clip_ids_order'] ok
    cur_clip_id = 0
    new_data['org_clip_ids_order'] = []
    if 'saliency_scores' in data:
        new_data['saliency_scores'] = []
    seg_secs = []
    for i in range(len(moment_segments)):

        # non-moment segment
        non_moment_segment = non_moment_segments[i]
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

        # moment segment
        moment_segment = moment_segments[i]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append((data['vid'], moment_segment['clip_id']))
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']
        seg_secs.append(moment_segment['seg_sec'])

    non_moment_segment = non_moment_segments[-1]
    new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()
        new_data['relevant_windows'] = find_ones_groups(new_clips, clip_len)
    else:
        new_data['relevant_windows'] = []
        for sup_m, sub_m in zip(find_ones_groups(new_clips, clip_len), seg_secs):
            sups, supe = sup_m
            subs, sube = sub_m
            new_data['relevant_windows'].append([sups + subs, supe - sube])


    ### Test ####
    if 'saliency_scores' in data:
        assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
        assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    clips_for_check = np.zeros(ctx_l)
    for _, (s, e) in new_data['org_clip_ids_order']:
        clips_for_check[s:e] += 1
    assert np.all(clips_for_check <= 1)
    assert np.all(clips_for_check[:-1] > 0)

    return new_data
