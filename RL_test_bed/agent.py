import tensorflow as tf
import os
from RL import model
from RL import architecture as policies
from RL import sonic_env as env
import gym

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def main():
    config = tf.ConfigProto()

    # Avoid warning message errors
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Allowing GPU memory growth
    config.gpu_options.allow_growth = True


    # Function to make an environment
    def make_env():
        return gym.make("Pong-v0")


    with tf.Session(config=config):
        model.learn(policy=policies.A2CPolicy,
                    env=SubprocVecEnv(
                        [make_env, make_env]),
                    nsteps=100,  # Steps per environment
                    total_timesteps=10000000,
                    gamma=0.99,
                    lam=0.95,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    lr=2e-4,
                    max_grad_norm=0.5,
                    log_interval=1
                    )


if __name__ == '__main__':
    main()
