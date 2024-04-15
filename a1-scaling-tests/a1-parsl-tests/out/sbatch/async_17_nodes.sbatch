#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/async.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/async.stderr
#SBATCH --nodes=17
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=chi150


source ~/setup_parsl_test_env.sh

for model in  1 18
do
    for workers in 2048
    do
	timeout 300 python test_async.py --config multinode --max_workers $workers --model $model --executor parsl &> async_$model.$workers.stdout
    done
done
