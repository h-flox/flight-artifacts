#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --account=chi150


source /home/yadunand/setup_parsl_test_env.v2.sh

for model in 3
do
    for workers in 2 4 8 16 32 64 128
    do
	timeout 300 python test_hack.py --config singlenode --max_workers $workers --model $model --executor parsl &> sync_$model.$workers.stdout
    done
done
