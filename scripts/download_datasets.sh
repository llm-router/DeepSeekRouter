mkdir -p ~/.cache/opencompass/
cd ~/.cache/opencompass/
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
opencompass --models hf_internlm2_5_1_8b_chat --datasets gpqa_gen --dry-run
opencompass --models hf_internlm2_5_1_8b_chat --datasets livecodebench_gen --dry-run
opencompass --models hf_internlm2_5_1_8b_chat --datasets aime2024_gen --dry-run

# wget https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA/resolve/main/chinese_simpleqa.jsonl