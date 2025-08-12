ckpt_path=$1
eval_split_name=test
eval_path=data/hl/highlight_test_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python la_tr_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
