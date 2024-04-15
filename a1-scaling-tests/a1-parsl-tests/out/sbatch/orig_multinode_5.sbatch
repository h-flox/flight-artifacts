#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=5
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=chi150


source /home/yadunand/setup_parsl_test_env.v2.sh
rm *rdb

for model in 3
do
    for workers in 512
    do
	timeout 300 python test_hack.py --config multinode --max_workers $workers --model $model --executor parsl &> sync_$model.$workers.stdout
    done
done
