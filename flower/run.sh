#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


echo "Starting server"
python server.py --clients $1 &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $1); do
    echo "Starting client $i"
    python client.py --partition-id 1 --model $2 --ip $3 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
