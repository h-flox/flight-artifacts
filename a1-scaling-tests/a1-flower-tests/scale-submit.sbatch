#!/bin/bash
#SBATCH --job-name="flower-test"
#SBATCH --output="flower-test-out.3.multi.out"
#SBATCH --partition=compute
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem=240G
#SBATCH --account=ind136
#SBATCH --export=ALL
#SBATCH -t 02:00:00

export NODES=$SLURM_JOB_NUM_NODES
ipad=$(hostname -I | awk '{print $3}')
echo $ipad

/bin/bash
/home/chard/miniconda3/bib/conda activate flower-test

for num_nodes in 2 4; do

	timeout 600 python /home/chard/flox-scaling-tests/a1-flower-tests/server.py --clients $((128 * $num_nodes)) --ip $ipad --model 3 &

	srun --ntasks $num_nodes -l --exclude=$SLURMD_NODENAME /home/chard/flox-scaling-tests/a1-flower-tests/run-client.sh 128 3 $ipad $num_nodes
	wait
done
