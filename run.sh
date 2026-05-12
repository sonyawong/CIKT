MODEL="Qwen/Qwen2.5-8B-Instruct"

python ./train_analyst.py \
    --profiles_dir ../student_simulation/cikt_profiles_recent_q_8_i_3 \
    --model_name_or_path $MODEL \
    --output_dir ./logs/ \
    --debug
