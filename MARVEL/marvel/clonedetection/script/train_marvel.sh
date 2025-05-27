save_name="marvel"
model_type="graphcodebert"

slip="_"
log_name=$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=6 nohup python ../code/run_mutual.py \
        --do_train \
        --do_test \
        --epochs=2 \
        --model_type $model_type \
        --save_name $save_name \
        --train_batch_size=4 \
        --eval_batch_size=4 \
        --alpha=0.3 \
        --max_adv_step=3 > ./train_$log_name.log 2>&1 &