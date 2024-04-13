#!/bin/bash

#SBATCH --job-name=Parsl.test
#SBATCH --output=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --error=/home/yadunand/flox-scaling-tests/parsl-tests/single_node.stdout
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --account=anl113


source ~/setup_parsl_test_env.sh

for model in 50
do
    for workers in 2 4 8 16 32 64 128
    do
	echo python test.py --workers_nodes $workers --model $model --executor parsl &> run_$model.$workers.stdout
    done
done
