dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=test-tr-detr
exp_id=exp

######## data paths
train_path=data/hl/highlight_train_release.jsonl
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
lr_drop=400
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2025
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4


results_root=results/la-tr-detr

gpunum=0

seed=2027

aug_seed=1



CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_tr_detr/train.py \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
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
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--train_path data/hl/hl_mmix_augseed_${aug_seed}.jsonl \
--exp_id exp_lad_mmix_${aug_seed}_seed_${seed} \
--m_classes "[13.80, 31, 75, 150]" \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--loss_m_classes "[10, 30, 150]" \
${@:1}
