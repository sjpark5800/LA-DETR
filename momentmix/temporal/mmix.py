from utils.basic_utils import save_jsonl, load_jsonl, l2_normalize_np_array, l2_normalize_torch_tensor
from copy import deepcopy
import random
import numpy as np
from fgmix import *
from bgmix import *
import sys
import torch
from run_on_video.data_utils import ClipFeatureExtractor
import torch.nn.functional as F
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = ClipFeatureExtractor(
    framerate=1, size=224, centercrop=True,
    model_name_or_path="ViT-B/32", device=device
)
        
def get_text_features(qid):

    q_feat_path = f"../features/clip_text_features/qid{qid}.npz"
    q_feat = np.load(q_feat_path)['pooler_output'].astype(np.float32)
    text_features = torch.tensor(q_feat)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features

def get_vid_features(vid, s, e):

    _feat_path = f"../features/clip_features/{vid}.npz"
    _feat = np.load(_feat_path)["features"].astype(np.float32)[s:e]
    image_features = torch.tensor(_feat)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.mean(dim=0)


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


savefilename += f"{dset_name}_mmix_{thres_crop}"
savefilename += f"_augseed_{seed}.jsonl"

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
                # moment_db[i].append((data['qid'], data['vid'], [start, end], get_vid_features(data['vid'], start, end)))
                break

    for start, end in non_moments:
        for i, db_range_value in enumerate(db_range):

            if (end-start) >= db_range_value:
                moment_db[i].append((data['qid'], data['vid'], [start, end]))
                break
            
print(f"Moment Database")
for db_range_, moment_db_ in zip(db_range, moment_db):
    print(f"moment db (>= {db_range_}) : {len(moment_db_)}")



##############################################
# If temporal relations exist, pass
##############################################

temporal_words = [
    # Sequential relationships (before, after)
    "before", "prior to", "earlier than", "previously", "formerly", "once", "ahead of", 
    "preceding", "by the time", "until", "after", "following", "subsequent to", "later", 
    "thereafter", "then", "next", "henceforth", "from then on", "since",

    # Simultaneous relationships (while, during)
    "while", "as", "when", "meanwhile", "simultaneously", "at the same time", "concurrently", 
    "in the meantime", "during", "throughout",

    # Continuous or overlapping relationships
    "until", "throughout", "all the while", "ever since", "as long as", "so long as", 
    "from start to finish",

    # Expressions emphasizing the passage of time
    "eventually", "gradually", "over time", "sooner or later", "in the long run", 
    "before long", "by then", "by now", "from now on", "hence"
]


# for qvhighlight dataset, remove temporal information from query
removed_temporal_info = {
    "Man in baseball cap eats before doing his interview.": "Man in a baseball cap eats.",
    "Camilla Cabello introduces her hair and makeup people 2 hours before the VMAs": "Camilla Cabello introduces her hair and makeup people.",
    "Girl looks at the water fountain before decided to drink from it.": "Girl looks at the water fountain.",
    "After a long ride through the countryside, a motorcycle rider arrives at a dirt parking lot with souvenir stands": "A motorcycle rider arrives at a dirt parking lot with souvenir stands.",
    "A woman sits at a heater to warm herself before a simple but superb dinner is served": "A woman sits at a heater to warm herself.",
    "Young African American couple taking precaution before on boarding": "Young African American couple taking precaution.",
    "A man is realeased from prison after decades inside and speaks to the public.": "A man is released from prison and speaks to the public.",
    "A guy getting his stuff into a car boot before his air travel": "A guy getting his stuff into a car boot for air travel.",
    "Family arrived three hours before and waiting to be onboard": "Family arrived and is waiting to be onboard.",
    "A guy applying a hair gel to his hair before he go out": "A guy applying hair gel to his hair.",
    "Two men walk in the wilderness before starting their interview.": "Two men walk in the wilderness.",
    "The young woman shows off her face after having treated her face with a facial.": "The young woman shows off her face.",
    "People talks about  devastation  after heavy flooding's in South India": "People talk about devastation in South India.",
    "A guy with a beard is talking before the camera.": "A guy with a beard is talking in front of the camera.",
    "After taking the lid off a pot of water on a gas heater, she puts the lid back on.": "She places a lid on a pot of water on a gas heater.",
    "A soldier is using a type of RPG before taking cover.": "A soldier is using a type of RPG.",
    "Soldier cries after seeing a dead child.": "A soldier cries.",
    "An investigative program lists other accidents in coal slurry impoundments since Buffalo Creek in 1972": "An investigative program lists other accidents in coal slurry impoundments.",
    "An anchor in a gray sweater is speaking after a clip of an avalanch beside arabic text.": "An anchor in a gray sweater is speaking beside Arabic text.",
    "A girl is taking pictures before the building.": "A girl is taking pictures in front of the building.",
    "After explaining how he purchased some train tickets, a man in a white hat and sunglasses has some lunch at a café.": "A man in a white hat and sunglasses has some lunch at a café.",
    "A picture of an official is shown before she gives a speech on stage.": "A picture of an official is shown. She gives a speech on stage.",
    "A group of women gather in a circle and hug before they discuss important matters.": "A group of women gather in a circle and hug. They discuss important matters.",
    "A woman checks her watch before showing her bagel sandwich": "A woman checks her watch and shows her bagel sandwich.",
    "After a woman speaks to congress, the galley holds up photos.": "A woman speaks to congress. The gallery holds up photos.",
    "After entering the kitchen, a woman begins to clean.": "A woman begins to clean the kitchen.",
    "White couple enjoying after skiing": "White couple enjoying themselves.",
    "After getting of his motorbike, the rider takes some photographs of the bike on his mobile phone.": "The rider takes some photographs of the bike on his mobile phone.",
    "After four women spend some time to take a good group picture, one of them burps": "Four women take a group picture, and one of them burps.",
    "After looking at parfaits, a couple see a ginormous omurice in a store window": "A couple see a ginormous omurice in a store window.",
    "A woman is petting a horse before riding it down a narrow pathway.": "A woman is petting a horse and riding it down a narrow pathway.",
    "A fluffy white cat rubs against a woman and eats after getting food in it's bowl.": "A fluffy white cat rubs against a woman and eats.",
    "A tired woman talks about all the things she still has to do before tomorrow": "A tired woman talks about all the things she still has to do.",
    "Kids excitement before boarding on a boat": "Kids show excitement for a boat.",
    "After finishing dinner, a woman starts on a delicious-looking dessert": "A woman starts on a delicious-looking dessert.",
    "A japanese restaurant chef talks about his philosophy as he cleans for an hour after closing": "A Japanese restaurant chef talks about his philosophy as he cleans.",
    "A guy looks tired after riding a cycle": "A guy looks tired.",
    "A man and a woman use a long stick to knock fruit out of a tree before eating it.": "A man and a woman use a long stick to knock fruit out of a tree and eat it.",
    "A bearded man in white sits before a mac laptop while talking.": "A bearded man in white sits before a Mac laptop while talking.",
    "People talk as they get dressed for the cold weather before going outside.": "People talk as they get dressed for the cold weather.",
    "A girl with dark grey top is brushing her hair after applying oil.": "A girl with a dark grey top is brushing her hair.",
    "A blue hallway leads to the plane and a view from the plane after takeoff.": "A blue hallway leads to the plane with a view from it.",
    "A couple is looking at a smaller private jet before boarding it and looking out the window during takeoff.": "A couple is looking at a smaller private jet, boarding it, and looking out the window.",
    "Man walks by graffiti before talking next to it.": "Man walks by graffiti and talks next to it.",
    "A muscular guy is shirtless and examining a sandwich he is holding before eating it.": "A muscular guy is shirtless and examining a sandwich he is holding.",
    "Weather reporter sharing some devastating scenes after storm": "Weather reporter sharing some devastating scenes.",
    "Footage of damaged vehicles after a flood": "Footage of damaged vehicles.",
    "After a window is smashed as an entry point, rioters enter the building and roam the hallways.": "Rioters enter the building and roam the hallways.",
    "A bunch of people are putting the bags in the car before go on a trip.": "A bunch of people are putting bags in the car for a trip.",
    "A girl is wearing a blue robe and showing her facial product before rubbing it on her face.": "A girl is wearing a blue robe and showing her facial product.",
    "A reporter in a blue surgical mask reporting next to the ruins of buildings after fires from a protests.": "A reporter in a blue surgical mask reporting next to the ruins of buildings.",
    "Weather reporter reports after hurricane LAURA devastation": "Weather reporter reports hurricane LAURA devastation.",
    "The after shot of blogger's video; a black screen with the blogger's Instagram handle is shown.": "A black screen with the blogger's Instagram handle is shown.",
    "President Trump is giving a speech and sitting before a board in a courtroom.": "President Trump is giving a speech and sitting in a courtroom.",
    "Man looks at the top shelf of his cabinets before putting things away.": "Man looks at the top shelf of his cabinets and puts things away.",
    "Woman shows off a bit of dinner before putting it on the plate.": "Woman shows off a bit of dinner and puts it on a plate.",
    "A couple people are walking and review a destroy city in ruins after a storm.": "A couple people are walking and review a destroyed city in ruins.",
    "Woman wears a helmet before she rides a atv.": "Woman wears a helmet and rides an ATV.",
    "Girl gives a tour of her hotel room before sitting down.": "Girl gives a tour of her hotel room and sits down.",
    "A man in teal is signing things for a crowd gathered before him.": "A man in teal is signing things for a crowd.",
    "A young woman talks about her walking exercise routine before walking to another room.": "A young woman talks about her walking exercise routine.",
    "After ordering drinks at Starbucks, 2 girls prepare them with sugar and stir it with straws.": "2 girls prepare Starbucks drinks with sugar and stir them with straws.",
    "A girl gets a piggy back ride from a boy in camo before standing side by side in a pink room.": "A girl gets a piggy back ride from a boy in camo, standing side by side in a pink room.",
    "Tow news presenters sit at a table and talk before a rocket is shown taking off.": "Two news presenters sit at a table and talk. A rocket is shown taking off.",
    "After some drinks, a woman makes her dance-challenged friend dance for her channel": "A woman makes her dance-challenged friend dance for her channel.",
    "Dog stands by the door until it goes for a walk.": "Dog stands by the door and goes for a walk.",
    "A woman in a pink dress is displaying her red shawl before putting it on.": "A woman in a pink dress is displaying her red shawl and putting it on.",
    "Man with black turban is sitting in a white car until he gets out of it.": "Man with a black turban is sitting in a white car and gets out.",
    "Man in blue suit and man with beige t shirt walk through the wilderness before the interview starts.": "Man in blue suit and man with a beige t shirt walk through the wilderness.",
    "A happy woman continues vlogging after dinner as the couple drink cups of chai": "A happy woman continues vlogging as the couple drink cups of chai.",
    "White couple giving review of tour before departing from spain": "White couple giving review of tour.",
    "Girl making a list before  shopping": "Girl making a list.",
    "A family is making the bed after waking up.": "A family is making the bed.",
    "A boy recovers in hospital after getting his tonsils taken out, his family surrounds him.": "A boy recovers in hospital, his family surrounds him.",
    "A young man showing his arm after being stung at the beach": "A young man showing his arm.",
    "A blonde woman is waiting in her seat on a plane before takeoff.": "A blonde woman is waiting in her seat on a plane.",
    "After many failures, a BMX biker finally lands a 360 flip off a near-vertical dirt mound": "A BMX biker finally lands a 360 flip off a near-vertical dirt mound.",
    "Stunning visuals of a blogger's outdoor shoot before boarding a car": "Stunning visuals of a blogger's outdoor shoot.",
    "An assortment of vegetables on a tray are before and after being in an oven.": "An assortment of vegetables on a tray.",
    "Hurricane effects after one years on echo system": "Hurricane effects on the ecosystem.",
    "Girls Expression after getting proposal": "Girls' expression at a proposal.",
    "Reporter reporting after heavy snow": "Reporter reporting heavy snow.",
    "A young woman gets soaked after playing water squirting with others": "A young woman gets soaked playing water squirting with others.",
    "A girl having dinner after a long day": "A girl having dinner.",
    "People stand by the pool in swimsuits before jumping in.": "People stand by the pool in swimsuits and jump in.",
    "Before driving to the hair salon, a woman in the driver's seat of her car rants to the camera.": "A woman in the driver's seat of her car rants to the camera.",
    "After being disappointed to find no seals at Seal Beach, a young woman spots some seals in the distance": "A young woman spots some seals in the distance.",
    "Woman walks around her hotel room after enterring it.": "Woman walks around her hotel room.",
    "Cyclists are waiting at the top of a tall stunt ramp before a competition.": "Cyclists are waiting at the top of a tall stunt ramp.",
    "A flight attendant gets a delicious fruit drink before her long day at work.": "A flight attendant gets a delicious fruit drink.",
    "A student takes a fried chicken lunch break before resuming her studies": "A student takes a fried chicken lunch break.",
    "After a long drive through a dry landscape, travellers reach a fishing destination": "Travellers reach a fishing destination.",
    "Official explain measures taken after extreme weather conditions": "An official explains measures taken.",
    "Nightmare before Christmas puppets are moving on a dark stage.": "Nightmare Before Christmas puppets are moving on a dark stage."
}

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




    # Consider Temporal relation #
    IsTempQuery = False
    for word in temporal_words:
        if word in data['query'].lower():
            IsTempQuery = True
    
    
    if not IsTempQuery:

        new_crop_data = fg_mix(data, moments=moments, non_moments=non_moments, thres_crop=thres_crop, ctx_l=ctx_l, clip_len=clip_len)
        if new_crop_data:
            new_datalist.append(new_crop_data)



new_datalist2 = []

for data in new_datalist:

    new_datalist2.append(deepcopy(data))


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
    
    
    # Consider Temporal relation #
    IsTempQuery = False
    for word in temporal_words:
        if word in data['query'].lower():
            IsTempQuery = True
    
    removeTemp = False
    if IsTempQuery:
        # before / after
        if data['query'] in removed_temporal_info:
            new_query = removed_temporal_info[data['query']]
            new_qid = (data['qid']) * 10000 + 1
            save_query_path = f'../features/clip_text_features/qid{new_qid}.npz'

            # Check if the file already exists
            if os.path.exists(save_query_path):
                print(f"File already exists: {save_query_path}")
            else:
                query_feat = feature_extractor.encode_text((new_query))[0]  # #text * (L, d)
                query_feat = F.normalize(query_feat, dim=0, eps=1e-5).cpu().numpy()

                np.savez(save_query_path, last_hidden_state=query_feat)
                print(f"Saved: {save_query_path}")
                removeTemp = True

    new_replace_data = bg_mix(data, moments=moments, non_moments=non_moments, ctx_l=ctx_l, clip_len=clip_len, db_range=db_range, moment_db=moment_db, text_feature = [])
    if new_replace_data:
        if removeTemp:
            new_crop_data['query'] = new_query
            new_crop_data['qid'] = new_qid
        new_datalist2.append(new_replace_data)



print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)} -> {len(new_datalist2)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist2, savefilename)
