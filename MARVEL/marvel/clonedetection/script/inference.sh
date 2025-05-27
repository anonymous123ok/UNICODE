attack_type="ALERT" # CODA ALERT CodeTAE
save_name="marvel"
model_type="graphcodebert" # graphcodebert  codebert codet5 codet5p
left=0
right=4000

slip="_"
log_name=$attack_type$slip$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=5 nohup  python ../code/inference.py \
    --attack_name=$attack_type \
    --model_type=$model_type \
    --eval_batch_size=8 \
    --index $left $right \
     --eval_data_file=/data/CODA_new_conrtainer/CODA_new/MARVEL_new/marvel/attacked_code/CloneDetection/${model_type}_${attack_type}_attacked_test_Code \
    --save_name=$save_name > ./final_defend_log/$log_name.log 2>&1 &


#    --eval_data_file=/data/CODA_new_conrtainer/CODA_new/MARVEL_new/marvel/attacked_code/CloneDetection/codebert_ALERT_attacked_test_Code \