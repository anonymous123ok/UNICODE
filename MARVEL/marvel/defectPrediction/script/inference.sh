attack_type="CodeTAE" # CODA ALERT CodeTAE
save_name="marvel"
model_type="codet5" # graphcodebert  codebert codet5 codet5p
left=0
right=10000

slip="_"
log_name=$attack_type$slip$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=4 python ../code/inference.py \
    --attack_name=$attack_type \
    --model_type=$model_type \
    --eval_batch_size=8 \
    --index $left $right \
    --eval_data_file=/data/CODA_new_conrtainer/CODA_new/MARVEL_new/marvel/attacked_code/C_Defects/${attack_type}_test_ori/${model_type}/test.txt \
    --save_name=$save_name > ./final_defend_log/$log_name.log 2>&1 &
