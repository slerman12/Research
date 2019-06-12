import time
import tensorflow as tf
import numpy as np
import os
from baselines.common.vec_env import SubprocVecEnv
from baselines import logger
from baselines.common import explained_variance
import gym

# Parameters
env_steps_per_update = 2048,  # Steps per environment for each update
total_steps = 10000000,
gamma = 0.99,
lam = 0.95,
vf_coef = 0.5,
ent_coef = 0.01,
lr = 2e-4,
max_grad_norm = 0.5,
log_interval = 10


# Function to make an environment
def make_env(env_idx):
    return gym.make("Pong-v0")


# Create environments
env = SubprocVecEnv([make_env, make_env])

# Avoid warning message errors
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config):
    noptepochs = 4

    # Batches are big, so how many times to divide them for training
    num_mini_batches = 8

    # Batch size is the number of environments by the number of steps each takes
    batch_size = env.num_envs * env_steps_per_update

    # How much to divide up the batches for rounds of training
    train_batch_size = batch_size // num_mini_batches
    assert batch_size % num_mini_batches == 0

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy,
                  ob_space=env.observation_space,
                  action_space=env.action_space,
                  nenvs=env.num_envs,
                  nsteps=env_steps_per_update,
                  ent_coef=ent_coef,
                  vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm)

    # Load the model to continue training
    load_path = "./Saved/"
    ckpt_name = "model.ckpt"
    model.load(load_path, ckpt_name)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=env_steps_per_update, total_timesteps=total_steps, gamma=gamma, lam=lam)

    # Start total timer
    time_first_start = time.time()

    # For each update batch in the total number of time steps
    for update in range(1, total_steps // batch_size + 1):
        # Start timer
        time_start = time.time()

        # Get minibatch
        obs, actions, returns, values = runner.run()

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mb_losses = []
        total_batches_train = 0

        # Index of each element of batch_size
        # Create the indices array
        indices = np.arange(batch_size)

        # Training
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(indices)

            # 0 to batch_size with train_batch_size step
            for start in range(0, batch_size, train_batch_size):
                end = start + train_batch_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values))
                mb_losses.append(model.train(*slices, lr))

        # Feedforward --> get losses --> update
        lossvalues = np.mean(mb_losses, axis=0)

        # End timer
        time_end = time.time()

        # Calculate the fps (frame per second)
        fps = int(batch_size / (time_end - time_start))

        if update % log_interval == 0 or update == 1:
            """
            Computes fraction of variance that ypred explains about y.
            Returns 1 - Var[y-ypred] / Var[y]
            interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
            """
            ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_steps", update * batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(time_end - time_first_start))
            logger.dump_tabular()

            savepath = "./models/" + str(update) + "/model.ckpt"
            model.save(savepath)
            print('Saving to', savepath)

    env.close()
