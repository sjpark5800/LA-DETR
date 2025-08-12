from copy import deepcopy
import random
import numpy as np
import math
from collections import defaultdict

def crop_clip_index(start_index, end_index, non_idx=False, num_crop=1, clip_len=2):

    if clip_len < 1:# vgg
        start_index = int(start_index) if start_index % 1 == 0 else int(start_index) + 1
        end_index = int(end_index)
        candidates = list(range((start_index) + 1, end_index, 1))
        num_crop = int(num_crop)
    else:
        candidates = list(range(start_index + clip_len, end_index, clip_len))

    if non_idx:
        candidates.append(-1) # not crop
    if num_crop > 1:
        return sorted(random.sample(candidates, num_crop))
    else: 
        return random.sample(candidates, num_crop)
    
def find_ones_groups(arr, clip_len):
    groups = []
    start_idx = None

    l = len(arr)
    for i in range(l):
        if arr[i] == 1 and start_idx is None:
            # 1이 처음 시작되는 인덱스 기록
            start_idx = i
        elif arr[i] == 0 and start_idx is not None:
            # 1의 그룹이 끝나는 인덱스 기록
            groups.append([start_idx * clip_len, i * clip_len])
            start_idx = None

    # 마지막 그룹이 배열 끝까지 이어지는 경우 처리
    if start_idx is not None:
        groups.append([start_idx * clip_len, len(arr) * clip_len])

    return groups


def find_zeros_groups(arr, clip_len):
    groups = []
    start_idx = None

    l = len(arr)
    for i in range(l):
        if arr[i] == 0 and start_idx is None:
            # 0이 처음 시작되는 인덱스 기록
            start_idx = i
        elif arr[i] == 1 and start_idx is not None:
            # 0의 그룹이 끝나는 인덱스 기록
            groups.append([start_idx * clip_len, i * clip_len])
            start_idx = None

    # 마지막 그룹이 배열 끝까지 이어지는 경우 처리
    if start_idx is not None:
        groups.append([start_idx * clip_len, len(arr) * clip_len])

    return groups






def merge_multi_moments(data, moments, non_moments, thres_merge, ctx_l, clip_len):

    ###############################################
    # 합칠 만한 short들 구하기
    ###############################################

    short_clip_idxs = []
    short_sum = 0
    for i, (s, e) in enumerate(moments):
        l = e - s
        if l <= 10:
            short_clip_idxs.append(i)
            short_sum += l

    # 합쳐서 short 이상이 안되면 이 data는 pass
    if short_sum <= thres_merge:
        return None

    ###############################################
    # short들은 합쳐서 새로운 moment segment 만들기
    ###############################################

    moment_segments = []

    short_clip_idx_ss = []

    ss_idx = 0
    for i, (s, e) in enumerate(moments):
        moment_len = (e - s) // clip_len
        ss_nxt_idx = ss_idx + moment_len

        if i not in short_clip_idxs:
            moment = dict()

            s_i, e_i = (s // clip_len if s != 0 else 0), e // clip_len
            moment['clip_id'] = [[s_i, e_i]]
            moment['len'] = (e - s) // clip_len
            moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]

            moment_segments.append(moment)

        else:
            short_clip_idx_ss.append((i, data['saliency_scores'][ss_idx : ss_nxt_idx]))

        ss_idx = ss_nxt_idx

    # short clips 섞기
    random.shuffle(short_clip_idx_ss)

    moment = dict()
    moment['clip_id'] = []
    moment['saliency_scores'] = []
    moment['len'] = 0
    for short_clip_idx, shrot_clip_ss in short_clip_idx_ss:
        s, e = moments[short_clip_idx]
        s_i, e_i = (s // clip_len if s != 0 else 0), e // clip_len

        moment['clip_id'].append([s_i, e_i])
        moment['len'] += (e - s) // clip_len
        moment['saliency_scores'] += shrot_clip_ss

    moment_segments.append(moment)


    ###############################################
    # non_moments들도 필요한 개수가 되도록 합쳐주기
    ###############################################

    # non_moments들을 랜덤으로 합치기
    need_merge_count = len(moment_segments) + 1

    # non_moment 무작위로 섞기
    random.shuffle(non_moments)

    non_moments_groups = []
    # 개수가 남는다면 합쳐주기
    if len(non_moments) > need_merge_count:
        # grounp boundary 뽑기
        group_sizes = sorted(random.sample(range(1, len(non_moments)), need_merge_count - 1))
        group_sizes.append(len(non_moments))

        prev_size = 0
        for size in group_sizes:
            non_moments_groups.append(non_moments[prev_size:size])
            prev_size = size

    # 개수가 딱 맞는다면 그대로 진행
    elif len(non_moments) == need_merge_count:
        for i, (s, e) in enumerate(non_moments):
            non_moments_groups.append([[s, e]])

    # 개수가 모자른다면 나눠주기
    else:
        need_crop_count = len(moment_segments) + 1 - len(non_moments)

        for i, (s, e) in enumerate(non_moments):
            l = e - s
            if l >= thres_merge * 2 and need_crop_count > 0:
                num_crop = min(l // 10 - 1, need_crop_count)
                
                ss = s
                for ee in crop_clip_index(s, e, num_crop=num_crop, clip_len=clip_len):
                    non_moments_groups.append([[ss, ee]])
                    ss = ee
                non_moments_groups.append([[ss, e]])

                need_crop_count -= num_crop
            else:
                non_moments_groups.append([[s, e]])

        # crop한 moment 사이에 끼워넣을 non-moment가 충분하지 않다면, 이 data는 pass
        if need_crop_count > 0 :
            return None

    ###############################################
    # non_moment_segments 만들기
    ###############################################

    non_moment_segments = []

    for non_moments_group in non_moments_groups:
        non_moment = dict()
        non_moment['clip_id'] = []
        non_moment['len'] = 0
        
        for i, (s, e) in enumerate(non_moments_group):
            non_moment['clip_id'].append([(s // clip_len if s != 0 else 0), e // clip_len])
            non_moment['len'] += (e - s) // clip_len

        non_moment_segments.append(non_moment)


    ###############################################
    # moment와 non-moment 섞기
    ###############################################

    random.shuffle(non_moment_segments)
    random.shuffle(moment_segments)



    ###############################################
    # 새로운 data 만들기
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
    new_data['saliency_scores'] = []
    for i in range(len(moment_segments)):

        # non-moment segment
        non_moment_segment = non_moment_segments[i]
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'] += non_moment_segment['clip_id']

        # moment segment
        moment_segment = moment_segments[i]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'] += moment_segment['clip_id']
        new_data['saliency_scores'] += moment_segment['saliency_scores']

    non_moment_segment = non_moment_segments[-1]
    cur_clip_id += non_moment_segment['len']
    new_data['org_clip_ids_order'] += non_moment_segment['clip_id']

    # new_data['relevant_clip_ids']
    new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()
    # new_data['relevant_windows']
    new_data['relevant_windows'] = find_ones_groups(new_clips, clip_len)

    #### Test 
    assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
    assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    clips_for_check = np.zeros(ctx_l)
    for s, e in new_data['org_clip_ids_order']:
        clips_for_check[s:e] += 1
    assert np.all(clips_for_check <= 1)
    assert np.all(clips_for_check > 0)

    return new_data

def merge_single_moment(data, moments, non_moments, thres_merge, ctx_l, clip_len):

    if 'and' in data['query']:
        return None
    
    s, e = moments[0]
    l = e - s

    if l <= thres_merge * 2 or l >= thres_merge * 30:
        return None


    ###############################################
    # long moment 쪼개고 short moment segment * 2 (copy) 만들기
    ###############################################

    num_crop = l // (thres_merge // 2) - 1
    num_crop = round(num_crop ** (1/2))

    moment_crop_idxs = crop_clip_index(s, e, num_crop=num_crop, clip_len=clip_len)
    moment_crop_idxs.append(e)

    moment_segments = []

    ss = s
    for ee in moment_crop_idxs:
        moment = dict()

        moment['clip_id'] = [int(ss // clip_len if ss != 0 else 0), int(ee // clip_len)]
        moment['len'] = int((ee - ss) // clip_len)

        if 'saliency_scores' in data:
            ss_nxt_idx = ss_idx + moment['len']
            moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
            ss_idx = ss_nxt_idx

        moment_segments.append(moment)
        moment_segments.append(deepcopy(moment))
        ss = ee

    random.shuffle(moment_segments)

    ###############################################
    # non_moments들 자르고 일부 누락시키기
    ###############################################

    non_moment_segments = []

    for s, e in non_moments:
        nl = e - s

        if nl <= thres_merge:
            return None
        
        num_crop = nl // (thres_merge // 2) - 1
        num_crop = round(num_crop ** (1/2))

        non_moment_crop_idxs = crop_clip_index(s, e, num_crop=num_crop, clip_len=clip_len)
        non_moment_crop_idxs.append(e)

        ss = s
        for ee in non_moment_crop_idxs:
            non_moment = dict()

            non_moment['clip_id'] = [int(ss // clip_len if ss != 0 else 0), int(ee // clip_len)]
            non_moment['len'] = int((ee - ss) // clip_len)

            non_moment_segments.append(non_moment)
            ss = ee

    random.shuffle(non_moment_segments)

    pop_l = 0
    inc_l = l // clip_len
    while pop_l < inc_l:
        pop_non_moment_segment = non_moment_segments.pop()
        pop_l += pop_non_moment_segment['len']
        if not non_moment_segments:
            return None

    ###############################################
    # 새로운 data 만들기
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['vid'] = data['vid']

    new_clips = np.zeros(ctx_l)

    cur_clip_id = 0
    new_data['org_clip_ids_order'] = []
    if 'saliency_scores' in data:
        new_data['saliency_scores'] = []

    non_moment_segments_len = len(non_moment_segments)
    fore_non_moment_segments = non_moment_segments[:non_moment_segments_len//2]
    back_non_moment_segments = non_moment_segments[non_moment_segments_len//2:]

    # fore - non moment
    for non_moment_segment in fore_non_moment_segments:
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append(non_moment_segment['clip_id'])

    # moment
    for moment_segment in moment_segments:
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append(moment_segment['clip_id'])
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']

    # back - non moment
    for non_moment_segment in back_non_moment_segments:
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append(non_moment_segment['clip_id'])

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()

    new_data['relevant_windows'] = find_ones_groups(new_clips, clip_len)
    new_data['duration'] = cur_clip_id * clip_len

    ########## Test

    if 'saliency_scores' in data:
        assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
        assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    clips_for_check = np.zeros(ctx_l)
    unique_order_clip_ids = list(map(list, set(map(tuple, new_data['org_clip_ids_order']))))
    for s, e in unique_order_clip_ids:
        clips_for_check[s:e] += 1
    assert np.all(clips_for_check <= 1)

    return new_data
