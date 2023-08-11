import argparse
import random
import time
import os
import math
import tqdm

# environment parameters
parser = argparse.ArgumentParser()
parser.add_argument("--env_id", default="simple_spread", type=str)
parser.add_argument("--train_episodes", default=50000, type=int)
parser.add_argument("--train_episode_length", default=25, type=int)
parser.add_argument("--test_freq", default=100, type=int)
parser.add_argument("--test_episodes", default=100, type=int)
parser.add_argument("--test_episode_length", default=25, type=int)
parser.add_argument("--test_times", default=5, type=int)
parser.add_argument("--agent_id", default="MADDPG", type=str, help="IPPO, MADDPG")
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--save_path", default="MADDPG", type=str)
config = parser.parse_args()
print(config)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
import numpy as np
import torch

from utils.make_env import make_env


def run(config):
    # create environment
    env = make_env(config.env_id, discrete_action=True)

    # select an agent (ps: random and random network need no training)
    if config.agent_id == "IPPO":
        from agents.IPPO.IPPO_agent import Agents as IPPOAgents
        agents = IPPOAgents(env.observation_space[0].shape[0], env.action_space[0].n)
    if config.agent_id == "MADDPG":
        from agents.MADDPG.MADDPG_agent import Agents as MADDPGAgents
        agents = MADDPGAgents(env.observation_space[0].shape[0], env.action_space[0].n)

    # train the agent
    print("----------Training Beginning----------")
    total_reward = 0.
    best_reward = -math.inf
    mean_reward_ls = []
    critic_loss_ls = []
    actor_loss_ls = []
    for ep_i in tqdm.tqdm(range(config.train_episodes)):
        obs = env.reset()
        episode_reward = 0.
        episode_length = 0.
        while episode_length < config.train_episode_length:
            episode_length += 1
            actions = agents.choose_action(obs)
            obs_, rewards, dones, infos = env.step(actions)
            critic_loss_sum, actor_loss_sum = agents.train(ep_i)
            agents.update_buffer(obs, actions, obs_, rewards, dones)
            obs = obs_
            episode_reward += np.array(rewards).sum()
        total_reward += episode_reward/episode_length

        # evaluate the agent
        if ep_i % config.test_freq == 0 or ep_i == config.train_episodes - 1:
            if ep_i == config.train_episodes - 1:
                print(f"Final testing in the last round!")
            mean_reward = 0
            for _ in range(config.test_times):
                mean_reward += evaluate(env, agents)
            mean_reward = mean_reward / config.test_times
            mean_reward_ls.append(mean_reward)
            print(f"Episode {ep_i}: mean_reward: {mean_reward}, critic_loss: {critic_loss_sum}, actor_loss: {actor_loss_sum}, best_reward: {best_reward}.")
            critic_loss_ls.append(critic_loss_sum.cpu().detach())
            actor_loss_ls.append(actor_loss_sum.cpu().detach())
            # save the best agent
            if mean_reward > best_reward:
                best_reward = mean_reward
                save_path = os.path.join("save", config.save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(agents, os.path.join(save_path, "agent.pth"))

    # save results
    save_path = os.path.join("save", config.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "rewards.npy"), np.array(mean_reward_ls))
    np.save(os.path.join(save_path, "critic_loss.npy"), np.array(critic_loss_ls))
    np.save(os.path.join(save_path, "actor_loss.npy"), np.array(actor_loss_ls))
    print(f"Best reward is {best_reward}")


# test the agent
def evaluate(env, agents):
    total_reward = 0.
    for ep_i in range(config.test_episodes):
        obs = env.reset()
        episode_reward = 0.
        for t_i in range(config.test_episode_length):
            actions = agents.act(obs)
            obs, rewards, dones, infos = env.step(actions)
            episode_reward += np.array(rewards).sum()
        total_reward += episode_reward/config.test_episode_length
    mean_reward = total_reward/config.test_episodes
    return mean_reward


if __name__ == '__main__':
    run(config)