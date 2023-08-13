#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2

cd ..

source ~/ts2vec/bin/activate

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
### M

python -u main_informer.py --model $model --data ETTh2 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ETTh2 --features M --seq_len 96 --label_len 96 --pred_len 48 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers $e_layers --d_layers 2 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers $e_layers --d_layers 2 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

python -u main_informer.py --model $model --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers $e_layers --d_layers 2 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool --data_aug $data_aug --cos_lr $cos_lr --learning_rate $learning_rate --freeze_encoder $freeze_encoder

### S

# python -u main_informer.py --model $model --data ETTh2 --features S --seq_len 48 --label_len 48 --pred_len 24 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --factor 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool

# python -u main_informer.py --model $model --data ETTh2 --features S --seq_len 96 --label_len 96 --pred_len 48 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool

# python -u main_informer.py --model $model --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 168 --e_layers $e_layers --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool

# python -u main_informer.py --model $model --data ETTh2 --features S --seq_len 336 --label_len 168 --pred_len 336 --e_layers $e_layers --d_layers 2 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool

# python -u main_informer.py --model $model --data ETTh2 --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers $e_layers --d_layers 2 --attn prob --des 'Exp' --itr 3 --batch_size 16 --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs $train_epoch --patience $patience --l2norm $l2norm --moco_average_pool $moco_average_pool
