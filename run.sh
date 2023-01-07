exp=roberta
max_length=128
model_dir=/home/v-chaozhang/gpt_detect/model/${exp}
log_dir=/home/v-chaozhang/gpt_detect/log/${exp}
model_name=roberta-base
#load_model_path=/home/v-chaozhang/gpt_detect/model/detector-base.pt


python main.py --max_length ${max_length} \
--model_dir ${model_dir} \
--log_dir ${log_dir} \
--model_name ${model_name} 
#--load_model_path ${load_model_path}