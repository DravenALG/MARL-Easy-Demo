import copy
import random

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.make_env import make_env

MADDPG_CONFIG = {}
MADDPG_CONFIG["ENV"] = make_env('simple_spread', discrete_action=True)
MADDPG_CONFIG["N_AGENT"] = 3
MADDPG_CONFIG["N_ACTION"] = MADDPG_CONFIG["ENV"].action_space[0].n  # 5
MADDPG_CONFIG["N_OBS"] = MADDPG_CONFIG["ENV"].observation_space[0].shape[0]  # 18
MADDPG_CONFIG["BUFFER_CAPACITY"] = 50000
MADDPG_CONFIG["BATCH_SIZE"] = 256
MADDPG_CONFIG["EPS"] = 0.05
MADDPG_CONFIG["EPS_MIN"] = 0.05
MADDPG_CONFIG["EPS_DECAY"] = (MADDPG_CONFIG["EPS"] - MADDPG_CONFIG["EPS_MIN"]) / 20000 / 25
MADDPG_CONFIG["EPS_DECAY"] = False
MADDPG_CONFIG["GAMMA"] = 0.95
MADDPG_CONFIG["LR_Critic"] = 1e-4
MADDPG_CONFIG["LR_Actor"] = 1e-4
MADDPG_CONFIG["SOFT_UPDATE"] = 1e-2
MADDPG_CONFIG['GRAD_CLIP'] = True
print(MADDPG_CONFIG)


# Critic
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim + act_dim, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        combined = torch.cat([obs, acts], 1)
        result1 = F.relu(self.FC1(combined))
        result2 = F.relu(self.FC2(result1))
        result3 = F.relu(self.FC3(result2))
        return self.FC4(result3)


# Actor
class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        result = (torch.tanh(self.FC4(result)) + 1) * 1.5
        return result


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, *transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        return self.buffer.clear()


class Agents:
    def __init__(self, n_obs, n_action):
        # agents
        actor1 = Actor(n_obs, n_action).cuda(0)
        actor2 = Actor(n_obs, n_action).cuda(0)
        actor3 = Actor(n_obs, n_action).cuda(0)

        # the central critic
        critic1 = Critic(MADDPG_CONFIG["N_AGENT"], n_obs, n_action).cuda(0)
        critic2 = Critic(MADDPG_CONFIG["N_AGENT"], n_obs, n_action).cuda(0)
        critic3 = Critic(MADDPG_CONFIG["N_AGENT"], n_obs, n_action).cuda(0)

        # actor optimizers
        actor_optim1 = torch.optim.Adam(actor1.parameters(), MADDPG_CONFIG["LR_Actor"])
        actor_optim2 = torch.optim.Adam(actor2.parameters(), MADDPG_CONFIG["LR_Actor"])
        actor_optim3 = torch.optim.Adam(actor3.parameters(), MADDPG_CONFIG["LR_Actor"])

        # critic optimizers
        critics_optim1 = torch.optim.Adam(critic1.parameters(), MADDPG_CONFIG["LR_Critic"])
        critics_optim2 = torch.optim.Adam(critic2.parameters(), MADDPG_CONFIG["LR_Critic"])
        critics_optim3 = torch.optim.Adam(critic3.parameters(), MADDPG_CONFIG["LR_Critic"])

        # replay buffer
        replay_buffer = ReplayBuffer(MADDPG_CONFIG["BUFFER_CAPACITY"])

        self.replay_buffer = replay_buffer
        self.actors = [actor1, actor2, actor3]
        self.critics = [critic1, critic2, critic3]
        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_optims = [actor_optim1, actor_optim2, actor_optim3]
        self.critics_optims = [critics_optim1, critics_optim2, critics_optim3]
        self.steps_done = 0

        self.loss_fn = nn.MSELoss()

    # add trajectory to the replay buffer
    def update_buffer(self, obs, actions, obs_, rewards, dones):
        self.replay_buffer.push(obs, actions, obs_, rewards, dones)

    def train(self, ep_i):
        critic_loss_sum = torch.tensor(0.).cuda()
        actor_loss_sum = torch.tensor(0.).cuda()

        if len(self.replay_buffer.buffer) > MADDPG_CONFIG["BATCH_SIZE"]:
            self.steps_done += 1
            for i in range(MADDPG_CONFIG["N_AGENT"]):
                obs, actions, obs_, rewards, _ = self.replay_buffer.sample(MADDPG_CONFIG["BATCH_SIZE"])
                obs = torch.Tensor(np.array(obs)).view(MADDPG_CONFIG["BATCH_SIZE"], -1).cuda(0)
                actions = torch.Tensor(np.array(actions)).view(MADDPG_CONFIG["BATCH_SIZE"], -1).cuda(0)
                obs_ = torch.Tensor(np.array(obs_)).view(MADDPG_CONFIG["BATCH_SIZE"], -1).cuda(0)
                rewards = torch.Tensor(np.array(rewards)).view(MADDPG_CONFIG["BATCH_SIZE"], -1).cuda(0)

                # calculate target_P and target_Q
                with torch.no_grad():
                    target_P = torch.Tensor().cuda(0)
                    for j in range(MADDPG_CONFIG["N_AGENT"]):
                        my_ob_ = obs_[:, j * MADDPG_CONFIG["N_OBS"]:(j + 1) * MADDPG_CONFIG["N_OBS"]]
                        my_target_P = self.actors_target[j](my_ob_)
                        target_P = torch.cat((target_P, my_target_P), dim=1)

                    target_Q = (rewards[:, i].view(MADDPG_CONFIG["BATCH_SIZE"], -1) + MADDPG_CONFIG["GAMMA"] *
                                self.critics_target[i](obs_, target_P)).detach()

                # update actor
                actor_loss_sum += self.update_actor(i, obs, actions)

                # update critic
                critic_loss_sum += self.update_critic(i, obs, actions, target_Q)

            if MADDPG_CONFIG["EPS_DECAY"]:
                MADDPG_CONFIG["EPS"] = MADDPG_CONFIG["EPS"] - (0.05 / 10000 / 25)
                if MADDPG_CONFIG["EPS"] < 0.05:
                    MADDPG_CONFIG["EPS"] = 0.05

            for i in range(MADDPG_CONFIG["N_AGENT"]):
                soft_update(self.critics_target[i], self.critics[i], MADDPG_CONFIG["SOFT_UPDATE"])
                soft_update(self.actors_target[i], self.actors[i], MADDPG_CONFIG["SOFT_UPDATE"])
            if MADDPG_CONFIG["EPS"] > MADDPG_CONFIG["EPS_MIN"]:
                MADDPG_CONFIG["EPS"] -= MADDPG_CONFIG["EPS_DECAY"]

        return critic_loss_sum, actor_loss_sum

    def update_critic(self, i, obs, actions, target_Q):
        v = self.critics[i](obs, actions)
        loss = self.loss_fn(v, target_Q)
        self.actor_optims[i].zero_grad()
        self.critics_optims[i].zero_grad()
        loss.backward()
        if MADDPG_CONFIG['GRAD_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 10)
        self.critics_optims[i].step()
        return loss

    def update_actor(self, i, obs, actions):
        my_ob = obs[:, i * MADDPG_CONFIG["N_OBS"]:(i + 1) * MADDPG_CONFIG["N_OBS"]]
        my_action = self.actors[i](my_ob)
        actions_tmp = copy.deepcopy(actions)
        actions_tmp[:, i * MADDPG_CONFIG["N_ACTION"]:(i + 1) * MADDPG_CONFIG["N_ACTION"]] = my_action
        loss = -self.critics[i](obs, actions_tmp).mean()
        self.actor_optims[i].zero_grad()
        self.critics_optims[i].zero_grad()
        loss.backward()
        if MADDPG_CONFIG['GRAD_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 10)
        self.actor_optims[i].step()
        return loss

    def choose_action(self, obs):
        if np.random.uniform() <= MADDPG_CONFIG["EPS"]:
            actions = []
            for i in range(MADDPG_CONFIG["N_AGENT"]):
                action = np.random.randint(0, MADDPG_CONFIG["N_ACTION"])
                action = np.eye(MADDPG_CONFIG["N_ACTION"])[action]
                actions.append(action)
        else:
            actions = self.act(obs)
        return actions

    def act(self, obs):
        torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1).cuda(0), requires_grad=False)
                     for i in range(MADDPG_CONFIG["N_AGENT"])]
        actions = []
        for i in range(MADDPG_CONFIG["N_AGENT"]):
            with torch.no_grad():
                action_continue = self.actors[i](torch_obs[i]).detach()
            actions.append(action_continue.detach().view(-1).cpu().numpy())
        return actions


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)
