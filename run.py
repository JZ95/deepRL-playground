import os
import gym
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from playground.register import agent_registry

os.environ['KMP_DUPLICATE_LIB_OK']='True'
tf.logging.set_verbosity(tf.logging.ERROR)

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 0.00001

BUFF_SIZE = 10000
BATCH_SIZE = 16
LR = 1e-4
GAMMA = 0.90

MAX_STEP = 10000000
LOG_STEP = 2000
SAVE_STEP = 10000
GRADIENT_STEP = 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=agent_registry.keys(), default='dqn', help="rl agent name implemented.")
    parser.add_argument("--game", type=str, default="LunarLander-v2", help="games supported by openAI-gym.")
    parser.add_argument("--mode", "-m", type=str, choices=["train", "play"], default="train")
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()
    if args.mode == "play" and args.ckpt is None:
        print("must offer a ckpt for playing.")
        exit()

    return args

def eps_annealing(step):
    x = 1. / (EPS_DECAY * step)
    return max(EPS_END, min(EPS_START, x))

def main():
    args = get_args()
    env = gym.make(args.game)
    observation = env.reset()

    agent = agent_registry[args.agent](LR, GAMMA, env.observation_space.shape, env.action_space.n)
    info = {'episode': 0, 'rewards': []}
    rewards = []

    for step in range(1, MAX_STEP):
        env.render()
        eps = eps_annealing(step) if args.mode == "train" else EPS_END
        if np.random.rand() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            q_vals = agent.eval(observation)
            action = np.argmax(q_vals)

        new_observation, reward, done, _ = env.step(action)
        agent.buffer((observation, action, reward, new_observation, done))
        observation = new_observation
        rewards.append(reward)

        if args.mode == "train":
            if step > 2 * BATCH_SIZE and step % GRADIENT_STEP == 0:
                agent.learn(BATCH_SIZE)

            if step % SAVE_STEP == 0:
                agent.save(step)

        if step % LOG_STEP == 0:
            msg = '[STEP: {step}, EPISODE: {episode}] ' \
                  'eps: {eps:.3f}, ' \
                  'avg_reward: {avg_reward:.3f}, ' \
                  'max_reward: {max_reward:.3f}'.format(step=step,
                                                        episode=info['episode'],
                                                        eps=eps,
                                                        avg_reward=np.mean(info['rewards'][-100:]),
                                                        max_reward=np.max(info['rewards'][-100:]))
            print(msg)

        if done:
            observation = env.reset()
            info['episode'] += 1
            info['rewards'].append(np.sum(rewards))
            rewards.clear()

    env.close()
    exit()


if __name__ == '__main__':
    main()