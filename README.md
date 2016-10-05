# cifar10-dist

An implementation of distributed cifar10 example that runs on TensorFlow.

## Run training program
'''
# Run a paramter server. The parameter server will block indefinitely. You can kill the process using "kill" command.
$ CUDA_VISIBLE_DEVICES='' python cifar10_dist_train.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=ps --task_id=0

# Run worker 0 in a different shell.
$ CUDA_VISIBLE_DEVICES='0' python cifar10_dist_train.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --max_steps=10000 --batch_size=128 --task_id=0

# Run worker 0 in a different shell.
$ CUDA_VISIBLE_DEVICES='1' python cifar10_dist_train.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --max_steps=10000 --batch_size=128 --task_id=1
'''
