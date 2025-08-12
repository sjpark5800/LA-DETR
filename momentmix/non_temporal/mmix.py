from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from fgmix import *
from bgmix import *
import sys


dset_name = sys.argv[1]
thres_crop = int(sys.argv[2])
seed = int(sys.argv[3])


print(f" ========== {dset_name} augmentation =============")
print(f" seed : {seed}")
print(f" thres_crop : {thres_crop}")




random.seed(seed)
np.random.seed(seed)

if dset_name == 'hl':
    datalist = load_jsonl('data/hl/highlight_train_release.jsonl')
    clip_len = 2
    db_range = [150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline

    savefilename = f"data/hl/"

    print(f" dset : QVHighlights")

elif dset_name == 'tacos':
    # tacos-train min: 0.4761904761904816 max: 751.4285714285714
    datalist = load_jsonl('data/tacos/train.jsonl')
    clip_len = 2
    db_range = [700, 500, 300, 200, 150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline

    savefilename = f"data/tacos/"

    print(f" dset : TACoS")

elif 'cha' in dset_name :
    # cha-train min: 1.6799999999999997 max: 80.80000000000001 
    datalist = load_jsonl('data/cha/charades_sta_train_tvr_format.jsonl')
    db_range = [80, 50, 30, 20, 10, 0] # moment class borderline

    savefilename = f"data/cha/"

    if 'vgg' in dset_name:
        clip_len = 0.166666
        print(f" dset : Charades - VGG")

    else:
        clip_len = 1
        print(f" dset : Charades")

elif 'nlq' in dset_name :
    datalist = load_jsonl('data/nlq/train.jsonl')
    clip_len = 2
    db_range = [500, 300, 200, 150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline

    savefilename = f"data/nlq/"

    print(f" dset : NLQ")

else:
    assert False


savefilename += f"{dset_name}_mmix_{thres_crop}.jsonl"

print(f" ============================================")


# another video moment database
moment_db = [[] for i in range(len(db_range))]

for data in datalist:

    ctx_l = int(data['duration'] // clip_len) if data['duration'] % clip_len == 0 else int(data['duration'] // clip_len) + 1

    if 'relevant_clip_ids' in data: # QVHighlights

        all_clips = np.zeros(ctx_l)
        all_clips[data['relevant_clip_ids']] = 1

        moments = find_ones_groups(all_clips, clip_len)
        assert moments == data['relevant_windows']

        non_moments = find_zeros_groups(all_clips, clip_len)

    else: # Charades, TACoS (single moment)
        moments = data['relevant_windows']
        non_moments = []
        if moments[0][0] != 0:
            non_moments.append([0, moments[0][0]])
        if moments[0][1] != data['duration']:
            non_moments.append([moments[0][1], data['duration']])    

    for start, end in moments:
        for i, db_range_value in enumerate(db_range):

            if (end-start) >= db_range_value:
                moment_db[i].append((data['qid'], data['vid'], [start, end]))
                break
            
print(f"Moment Database")
for db_range_, moment_db_ in zip(db_range, moment_db):
    print(f"moment db (>= {db_range_}) : {len(moment_db_)}")


new_datalist = []

for data in datalist:

    new_datalist.append(deepcopy(data))

    ctx_l = int(data['duration'] // clip_len) if data['duration'] % clip_len == 0 else int(data['duration'] // clip_len) + 1

    ###############################################
    # Get moment and non-moment segments
    ###############################################

    if 'relevant_clip_ids' in data: # QVHighlights

        all_clips = np.zeros(ctx_l)
        all_clips[data['relevant_clip_ids']] = 1

        moments = find_ones_groups(all_clips, clip_len)
        assert moments == data['relevant_windows']

        non_moments = find_zeros_groups(all_clips, clip_len)

    else: # Charades, TACoS (single moment)
        moments = data['relevant_windows']
        non_moments = []
        if moments[0][0] != 0:
            non_moments.append([0, moments[0][0]])
        if moments[0][1] != data['duration']:
            non_moments.append([moments[0][1], data['duration']])    

    # If no non-moments exist, this data cannot be used
    if not non_moments:
        continue 
    
    # crop augmentation
    new_crop_data = fg_mix(data, moments=moments, non_moments=non_moments, thres_crop=thres_crop, ctx_l=ctx_l, clip_len=clip_len)
    if new_crop_data:
        new_datalist.append(new_crop_data)

    new_replace_data = bg_mix(data, moments=moments, non_moments=non_moments, ctx_l=ctx_l, clip_len=clip_len, db_range=db_range, moment_db=moment_db)
    if new_replace_data:
        new_datalist.append(new_replace_data)

print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist, savefilename)
