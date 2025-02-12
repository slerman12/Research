import datetime
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

    name = "pong"

    dt = datetime.datetime.now()
    os.environ['OPENAI_LOGDIR'] = "./Logs/{}/{}_{}_{}_{}_{}/".format(name, dt.year, dt.month, dt.day, dt.hour, dt.minute)

    # Function to make an environment
    def make_env():
        e = gym.make("Pong-v0")
        e = env.PreprocessFrame(e)
        return env.FrameStack(e, 4)

    with tf.Session(config=config):
        model.learn(policy=policies.A2CPolicy,
                    # env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3,
                    #                    env.make_train_4, env.make_train_5,env.make_train_6,env.make_train_7,
                    #                    env.make_train_8,env.make_train_9,env.make_train_10,env.make_train_11,
                    #                    env.make_train_12 ]),
                    env=SubprocVecEnv(
                        [make_env, make_env, make_env, make_env, make_env, make_env, make_env, make_env, make_env,
                         make_env, make_env, make_env]),
                    # nsteps=2048,  # Steps per environment
                    nsteps=5,  # Steps per environment
                    total_timesteps=40000000,
                    gamma=0.99,
                    lam=0.95,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    lr=2e-4,
                    max_grad_norm=0.5,
                    log_interval=100,
                    name=name
                    )


if __name__ == '__main__':
    main()
