ckpt_path=result_1109_fin/qv/hl-video_tef-lad_tempandfeat_5_1_seed_2021//model_best.ckpt
eval_split_name=test
eval_path=data/highlight_${eval_split_name}_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python uvcom/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
