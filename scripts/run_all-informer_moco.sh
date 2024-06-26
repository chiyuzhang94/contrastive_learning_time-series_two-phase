weight=0.5
model=tcn_moco
mask_rate=0.3
des="./twophase_res"
train_epoch=20
patience=5
l2norm="True"
moco_average_pool="False"
data_aug="cost"
cos_lr="True"
e_layers=5
freeze_encoder="False"
learning_rate=0.001
mare="False"

for task in ETTh1 ETTm1 ECL 
    do
    for model in 'informer-moco'
        do
            echo $task$model
            sbatch --time=120:0:0 --output=./out_file/"$task""$model"_l2"$l2norm"_avg${moco_average_pool}_data${data_aug}_cos${cos_lr}_freeze${freeze_encoder}.out --job-name="$task""$model" "$task"-hyper.sh $mask_rate $weight $model $des $train_epoch $patience $l2norm $moco_average_pool $data_aug $cos_lr $e_layers $learning_rate $freeze_encoder $mare
            sleep 10
        done
done
