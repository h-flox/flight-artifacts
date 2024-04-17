README
======


Please install all dependencies into a conda environment with:

```
conda create --name parsl_py3.11 python=3.11
conda activate parsl_py3.11
echo "conda activate parsl_py3.11" > ~/setup_parsl_test_env.sh
(parsl_py3.10) pip install -r requirements.txt
(parsl_py3.10) pip install --no-deps pydantic==2.6.3
```

Experiment 1: FloX with Parsl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a single instance of the experiment on a singenode to sanity-check

``` bash
# model 3 is SqueezeNet, singlenode launches workers on the same node
python test.py --config singlenode --max_workers 2 --model 0 --executor parsl
```

To launch the complete experiment on SDSC Expanse2

``` bash
bash launch_parsl.sh
```

Experiment 2: Flox with Parsl+RedisConnector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To launch the experiment with the Parsl backed with RedisConnector for data transfer
follow the steps below:

``` bash
redis-server /home/yadunand/flox-scaling-tests/parsl-tests/redis.conf &
python test.py --config multinode --max_workers 4 --model 3 --executor parsl
```

Launch the complete set of experiments on SDSC Expanse2

``` bash
bash launch_parsl_redisconnector.sh
```