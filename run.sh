exp=gpt2-pretrain
max_length=128
model_dir=/home/v-chaozhang/gpt_detect/model/${exp}
log_dir=/home/v-chaozhang/gpt_detect/log/${exp}
model_name=roberta-base
tokenizer_name=roberta-base
trainer=KfoldTrainer
dataset=Kfold_GPTDataset
load_model_path=/home/v-chaozhang/gpt_detect/model/detector-base.pt


python main.py --max_length ${max_length} \
--model_dir ${model_dir} \
--log_dir ${log_dir} \
--model_name ${model_name} \
--trainer ${trainer} \
--dataset ${dataset} \
--tokenizer_name ${tokenizer_name} \
--load_model_path ${load_model_path} 