#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=3
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=chi150


source ~/setup_parsl_test_env.sh

for model in 0 1 3 18 50 152
do
    for workers in 256
    do
	timeout 600 python test.py --config multinode --max_workers $workers --model $model --executor parsl &> sync_$model.$workers.stdout
    done
done
