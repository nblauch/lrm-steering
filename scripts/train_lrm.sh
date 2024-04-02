declare -a alphas=(0.5 0.9 1)

for alpha in ${alphas[@]}; do
    COMMAND="python scripts/train_lrm.py --config config/lrm3.yaml --training.distributed 0 --model.alpha ${alpha} --training.loss supervised --model.mlp 4096-4096-1000 --training.batch_size 2048"

    echo $COMMAND

    # $COMMAND

    sbatch --export="COMMAND=$COMMAND" --job-name lrm -p kempner --cpus-per-task=12 --gres=gpu:nvidia_a100-sxm4-40gb:1 --mem=256GB --time 24:00:00 --output=log/%j.log scripts/run_slurm.sbatch
done

