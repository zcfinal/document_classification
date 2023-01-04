exp=debug
max_length=30
model_dir=/home/v-chaozhang/mind_data/model/${exp}
log_dir=/home/v-chaozhang/mind_data/log/${exp}

python main.py --max_length ${max_length} \
--model_dir ${model_dir} \
--log_dir ${log_dir}