# exp=tfidf
# max_length=128
# model_dir=../model/${exp}
# log_dir=../log/${exp}
# model_name=logistic
# tokenizer_name=tf-idf
# trainer=KfoldTFIDFTrainer
# dataset=Kfold_TFIDFDataset

# python main.py --max_length ${max_length} \
# --model_dir ${model_dir} \
# --log_dir ${log_dir} \
# --model_name ${model_name} \
# --trainer ${trainer} \
# --dataset ${dataset} \
# --tokenizer_name ${tokenizer_name} \
# # --question_feature \


exp=gpt2-pretrain
max_length=128
model_dir=../model/${exp}
log_dir=../log/${exp}
model_name=roberta-base
tokenizer_name=roberta-base
trainer=KfoldTrainer
dataset=Kfold_GPTDataset
load_model_path=/home/v-derongxu/others/gpt-2-output-dataset/detector-base.pt


python main.py --max_length ${max_length} \
--model_dir ${model_dir} \
--log_dir ${log_dir} \
--model_name ${model_name} \
--trainer ${trainer} \
--dataset ${dataset} \
--tokenizer_name ${tokenizer_name} \
# --question_feature \
--load_model_path ${load_model_path} 