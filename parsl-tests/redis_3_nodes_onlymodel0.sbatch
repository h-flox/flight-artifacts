#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=3
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=chi150


source ~/setup_parsl_test_env.sh
rm *rdb
# Disable auth and bind to all interfaces
redis-server /home/yadunand/flox-scaling-tests/parsl-tests/redis.conf &
    
for model in 0
do
    for workers in 256
    do
	timeout 200 python test.py --config multinode --max_workers $workers --model $model --executor parsl &> redis_$model.$workers.stdout
    done
done
