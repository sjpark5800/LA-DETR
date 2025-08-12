dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/hl/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32

results_root=results/la-qd-detr

gpunum=1

seed=2023

aug_seed=0


# Apple both MMix and LAD
CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--train_path data/hl/hl_mmix_augseed_${aug_seed}.jsonl \
--exp_id exp_lad_mmix_${aug_seed}_seed_${seed} \
--m_classes "[13.8, 32.0, 55.0, 150]" \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--loss_m_classes "[10, 30, 150]" \
${@:1}




# # Apply only MMix
# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --train_path data/hl/hl_mmix_augseed_${aug_seed}.jsonl \
# --exp_id exp_mmix_${aug_seed}_seed_${seed} \
# --seed ${seed} \
# --loss_m_classes "[10, 30, 150]" \
# ${@:1}


# # Apply only LAD
# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_qd_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --train_path ${train_path} \
# --exp_id exp_lad_seed_${seed} \
# --m_classes "[13.8, 32.0, 55.0, 150]" \
# --tgt_embed \
# --cc_matching \
# --seed ${seed} \
# --loss_m_classes "[10, 30, 150]" \
# ${@:1}
