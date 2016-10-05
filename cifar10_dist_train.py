# coding: utf-8
# Copyright 2016 Joongsoo Lee @ ETRI. All Rights Reserved.
#
# =============================================================================
"""Cifar-10을 사용한 멀티 노드 트레이닝

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:2222,machine3:2222""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g."""
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
# The interval seconds are not recommended values.
# For example, in case of cifar10 multi-GPU version, summary strings are saved
# in every 100 steps and checkpoints are saved in every 1000 steps.
# In case of Inception_v3, save_interval_secs and save_summaries_secs are
# 600 and 180 respectively.
tf.app.flags.DEFINE_integer('save_interval_secs', 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 20,
                            'Save summaries interval seconds.')


def train(target, cluster_spec):
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])

    if FLAGS.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

    assert num_workers > 0 and num_parameter_servers > 0, \
        ('num_workers and num_parameter_servers must be > 0.')

    is_chief = (FLAGS.task_id == 0)

    with tf.device(
        tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster_spec)):

            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     FLAGS.batch_size)
            decay_steps = \
                int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

            # Learning rate decay.
            lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            cifar10.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)

            tf.scalar_summary('learning_rate', lr)

            opt = tf.train.GradientDescentOptimizer(lr)

            images, labels = cifar10.distorted_inputs()

            logits = cifar10.inference(images)

            cifar10.loss(logits, labels)

            losses = tf.get_collection('losses')

            total_loss = tf.add_n(losses, name='total_loss')

            if is_chief:

                loss_averages = tf.train.ExponentialMovingAverage(
                    0.9, name='avg')
                loss_averages_op = loss_averages.apply(losses + [total_loss])

                for l in losses + [total_loss]:
                    loss_name = l.op.name
                    # Name each loss as '(raw)' and name the moving average
                    # version of the loss as the original loss name.
                    tf.scalar_summary(loss_name + ' (raw)', l)
                    tf.scalar_summary(loss_name, loss_averages.average(l))

                with tf.control_dependencies([loss_averages_op]):
                    total_loss = tf.identity(total_loss)

            exp_moving_averager = tf.train.ExponentialMovingAverage(
                cifar10.MOVING_AVERAGE_DECAY, global_step)

            variables_to_average = (
                tf.trainable_variables() + tf.moving_average_variables())

            for var in variables_to_average:
                tf.histogram_summary(var.op.name, var)

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_replicas_to_aggregate,
                replica_id=FLAGS.task_id,
                total_num_replicas=num_workers,
                variable_averages=exp_moving_averager,
                variables_to_average=variables_to_average)

            grads = opt.compute_gradients(total_loss)

            for grad, var in grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            apply_gradients_op = opt.apply_gradients(
                grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')

            # Get chief queue_runners, init_tokens and clean_up_op, which is
            # used to synchronize replicas.
            # More details can be found in sync_replicas_optimizer.
            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()
            clean_up_op = opt.get_clean_up_op()

            # Create a saver.
            saver = tf.train.Saver()

            # Build the summary operation based on the TF collection of
            # Summaries.
            summary_op = tf.merge_all_summaries()

            # Build an initialization operation to run below.
            init_op = tf.initialize_all_variables()

            # We run the summaries in the same thread as the training
            # operations by passing in None for summary_op to avoid
            # a summary_thread being started. Running summaries and training
            # operations in parallel could run out of GPU memory.
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=init_op,
                                     summary_op=None,
                                     global_step=global_step,
                                     saver=saver,
                                     save_model_secs=FLAGS.save_interval_secs)

            tf.logging.info('%s Supervisor' % datetime.now())

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)

            sess = sv.prepare_or_wait_for_session(target, config=sess_config)

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
            tf.logging.info('Started %d queues for processing input data.',
                            len(queue_runners))

            if is_chief:
                sv.start_queue_runners(sess, chief_queue_runners)
                sess.run(init_tokens_op)

            # Train, checking for Nans. Concurrently run the summary operation at a
            # specified interval. Note that the summary_op and train_op never run
            # simultaneously in order to prevent running out of GPU memory.
            next_summary_time = time.time() + FLAGS.save_summaries_secs
            while not sv.should_stop():
                try:
                    start_time = time.time()
                    # TODO should check global_step
                    loss_value, step = sess.run([train_op, global_step])
                    # loss_value, step = sess.run([train_op])
                    # print(step)
                    assert not np.isnan(loss_value), 'Model diverged (loss = NaN)'
                    if step > FLAGS.max_steps:
                        break
                    duration = time.time() - start_time

                    if step % 10 == 0:
                        examples_per_sec = FLAGS.batch_size / float(duration)
                        format_str = ('Worker %d: %s: step %d, loss = %.2f'
                                      '(%.1f examples/sec; %.3f  sec/batch)')
                        tf.logging.info(format_str %
                                        (FLAGS.task_id, datetime.now(), step,
                                         loss_value, examples_per_sec, duration))

                    if is_chief and next_summary_time < time.time():
                        tf.logging.info('Running Summary operation on the chief.')
                        summary_str = sess.run(summary_op)
                        sv.summary_computed(sess, summary_str)
                        tf.logging.info('Finished running Summary operation.')

                        next_summary_time += FLAGS.save_summaries_secs

                except:
                    if is_chief:
                        tf.logging.info('About to execute sync_clean_up_op!')
                        sess.run(clean_up_op)
                    raise

            # sv.stop()
            # sv.request_stop()

            if is_chief:
                saver.save(sess,
                           os.path.join(FLAGS.train_dir, 'model.ckpt'),
                           global_step=global_step)


def main(unused_args):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
         'worker': worker_hosts},
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        # Only the chief checks for or creates train_dir.
        if FLAGS.task_id == 0:
            if not tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.MakeDirs(FLAGS.train_dir)
        train(server.target, cluster_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
