#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --account=chi150


source ~/setup_parsl_test_env.sh
for model in  0 1 18 50 152
do
    for workers in 256
    do
	timeout 200 python test.py --config multinode --max_workers $workers --model $model --executor parsl &> run_multinode_$model.$workers.stdout
    done
done
