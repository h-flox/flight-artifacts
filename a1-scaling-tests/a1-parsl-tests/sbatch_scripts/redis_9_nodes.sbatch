#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=9
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=chi150


source ~/setup_parsl_test_env.sh

# Disable auth and bind to all interfaces
redis-server /home/yadunand/flox-scaling-tests/parsl-tests/redis.conf &
rm *rdb

for model in 0 1 3 18 50 152
do
    for workers in 1024
    do
	timeout 1200 python test_redis.py --config multinode --max_workers $workers --model $model --executor parsl &> redis_$model.$workers.stdout
    done
done
