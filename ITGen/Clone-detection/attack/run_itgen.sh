left=0
right=500
#conda install -c conda-forge libstdcxx-ng libgcc-ng
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=2 nohup python attack_itgen.py \
        --left_index  ${left} \
        --output_dir=../saved_models \
        --model_type=roberta \
        --tokenizer_name=/root/CODA/microsoft/codebert-base \
        --model_name_or_path=/root/CODA/microsoft/codebert-base \
        --csv_store_path ./result/attack_itgen_${left}_${right}.jsonl \
        --base_model=/root/CODA/microsoft/codebert-base-mlm \
        --eval_data_file=/workspace/CODA_new/CODA/test/CloneDetection/dataset/test_sampled_${left}_${right}.txt \
        --block_size 512 \
        --eval_batch_size 2 \
        --seed 123456 > ${left}.log &