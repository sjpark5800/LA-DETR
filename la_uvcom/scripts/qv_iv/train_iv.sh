# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1

dset_name=hl
ctx_mode=video_tef
v_feat_types=internvideo
t_feat_type=llama 
device=1
enc_layers=3
dec_layers=3
query_num=10
n_txt_mu=5
n_visual_mu=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2023
lr=1e-4
lr_gamma=0.1
neg_choose_epoch=80
lr_drop=100

######## data paths
train_path=data/hl_iv/highlight_train_release_internvideo2.jsonl
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
if [[ ${v_feat_types} == *"internvideo"* ]]; then
  v_feat_dirs+=(${feat_root}/qvhighlight_6b)
  (( v_feat_dim += 768 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
fi
if [[ ${t_feat_type} == *"llama"* ]]; then
  t_feat_dir+=(${feat_root}/qvhighlight_llama_text_feature)
  t_feat_dim=4096
fi

#### training
bsz=32


gpunum=0

results_root=results/la-uvcom/qv-iv


list="2021 2022 2023 2024 2025 2026 2027"

for seed in $list
do
  echo $seed

aug_seed=0
CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_uvcom/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--train_path data/hl_iv/hl_iv_mmix_5_augseed_${aug_seed}.jsonl \
--exp_id exp_lad_mmix_${aug_seed}_seed_${seed} \
--m_classes "[13.4, 31.5, 75.0, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
${@:1}

aug_seed=1
CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_uvcom/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--train_path data/hl_iv/hl_iv_mmix_5_augseed_${aug_seed}.jsonl \
--exp_id exp_lad_mmix_${aug_seed}_seed_${seed} \
--m_classes "[13.4, 31.5, 75.0, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
${@:1}

done