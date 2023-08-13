#!/bin/bash
cd ..

mask_rate=$1
losslambda=$2
model=$3
des=$4
train_epoch=$5
patience=$6
l2norm=$7
moco_average_pool=$8
data_aug=$9
cos_lr=${10}
e_layers=${11}
learning_rate=${12}
freeze_encoder=${13}
svm_evaluate=${14}
mare=${15}

### M

python -u main_informer.py --model $model --data ECL --features M --e_layers $e_layers --attn prob --des 'Exp' --itr 3 --factor 3 --batch_size 16 --root_path './data/' --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --freq 'h' --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder --svm_evaluate $svm_evaluate --mare $mare

python -u main_informer.py --model $model --data ECL --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --root_path './data/' --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --freq 'h' --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ECL --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --root_path './data/' --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --freq 'h' --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ECL --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --root_path './data/' --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --freq 'h' --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder #seq_len is different to orignial informer

python -u main_informer.py --model $model --data ECL --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --root_path './data/' --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --freq 'h' --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder #seq_len is different to orignial informer