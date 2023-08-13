weight=0.5
model=tcn_moco
mask_rate=0.5
des="./svm_res/"
train_epoch=20
patience=5
l2norm="True"
moco_average_pool="False"
data_aug="cost"
cos_lr="True"
e_layers=10
freeze_encoder="True"
learning_rate=0.001
svm_evaluate="True"
mare="False"

for task in ETTh1 ETTm1 ECL
    do
    for model in 'dtcn-moco' 'tcn-moco'
        do
            echo $task$model
            sbatch --time=120:0:0 --output=./out_file/"$task""$model"_l2"$l2norm"_avg${moco_average_pool}_data${data_aug}_cos${cos_lr}_svm.out --job-name="$task""$model" "$task"-svm.sh $mask_rate $weight $model $des $train_epoch $patience $l2norm $moco_average_pool $data_aug $cos_lr $e_layers $learning_rate $freeze_encoder $svm_evaluate $mare
            sleep 10
        done
done
