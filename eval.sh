# 正式评估
# conda run -n llmkt python cikt/evaluate_analyst.py \
#     --lora_weights cikt/analyst_ckpt/lora_weights \
#     --base_model Qwen/Qwen2.5-7B-Instruct \
#     --output_dir cikt/eval_results

MODEL="Qwen/Qwen2.5-8B-Instruct"

# # debug（2条）
# python ./evaluate_analyst.py \
#     --lora_weights ./analyst_ckpt/lora_weights \
#     --base_model $MODEL \
#     --output_dir ./analyst_eval --debug

# # 对比 base model（不加 --lora_weights）
# conda run -n llmkt python cikt/evaluate_analyst.py \
#     --base_model $MODEL \
#     --output_dir cikt/eval_results_base
