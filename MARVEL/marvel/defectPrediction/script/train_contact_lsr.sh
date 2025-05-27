save_name="marvel"
model_type="codet5"

slip="_"
log_name=$model_type$slip$save_name

CUDA_VISIBLE_DEVICES=7 nohup python ../code/run_contact_lsr.py \
        --do_train \
        --epochs=10 \
        --model_type $model_type \
        --save_name $save_name \
        --train_batch_size=8 \
        --eval_batch_size=8 \
        --alpha=0.3 \
        --max_adv_step=3 > ./train_${log_name}_cont_lsr.log 2>&1 &