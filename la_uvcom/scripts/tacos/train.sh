# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1

dset_name=tacos
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
device=1
enc_layers=3
dec_layers=3
query_num=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2018
lr=1e-4
lr_gamma=0.1
clip_length=2
neg_choose_epoch=70
lr_drop=100
max_v=-1

######## data paths
train_path=data/tacos/train.jsonl
eval_path=data/tacos/val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features/tacos

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
bsz=16

gpunum=2

results_root=results/la-uvcom/tacos

seed=2023


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
--clip_length ${clip_length} \
--neg_choose_epoch ${neg_choose_epoch} \
--lr_drop ${lr_drop} \
--max_v_l ${max_v} \
--dec_layers ${dec_layers} \
--num_queries 10 \
--seed ${seed} \
--train_path data/tacos/tacos_mmix_10.jsonl \
--exp_id exp_lad_mmix_seed_${seed} \
--m_classes "[10.25, 19.34, 38.34, 10000000]" \
--no_text \
--cc_matching \
--tgt_embed \
--n_epoch 200 \
${@:1}


_eval_split_name=val
_eval_path=data/tacos/test.jsonl

echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_uvcom/inference.py \
--resume results/la-uvcom/tacos/tacos-video_tef-exp_lad_mmix_seed_${seed}/model_best.ckpt \
--eval_split_name ${_eval_split_name} \
--eval_path ${_eval_path} \
${@:3}



