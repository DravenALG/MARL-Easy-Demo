import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.make_env import make_env

IPPO_CONFIG = {}
IPPO_CONFIG["ENV"] = make_env('simple_spread', discrete_action=True)
IPPO_CONFIG["N_AGENT"] = 3
IPPO_CONFIG["N_ACTION"] = IPPO_CONFIG["ENV"].action_space[0].n  # 5
IPPO_CONFIG["N_OBS"] = IPPO_CONFIG["ENV"].observation_space[0].shape[0]  # 18
IPPO_CONFIG["BUFFER_CAPACITY"] = 50000
IPPO_CONFIG["EPSILON"] = 0.2
IPPO_CONFIG["EPS"] = 0.01
IPPO_CONFIG["GAMMA"] = 0.99
IPPO_CONFIG["LR_Critic"] = 5e-4
IPPO_CONFIG["LR_Actor"] = 5e-4
IPPO_CONFIG["N_UPDATE"] = 3
IPPO_CONFIG['GRAD_CLIP'] = True
print(IPPO_CONFIG)


class Actor(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, X):
        X = torch.Tensor(X)
        h1 = F.relu(self.fc1(X))
        h2 = F.relu(self.fc2(h1))
        out = (torch.tanh(self.fc3(h2)) + 1) * 1.5
        return out


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, nonlin=F.relu):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.nonlin = nonlin

    def forward(self, X):
        X = torch.Tensor(X)
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, *transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self):
        return zip(*self.buffer)

    def clean(self):
        return self.buffer.clear()


class Agents:
    def __init__(self, n_obs, n_action):
        # agents
        actor1 = Actor(n_obs, n_action).cuda(0)
        actor2 = Actor(n_obs, n_action).cuda(0)
        actor3 = Actor(n_obs, n_action).cuda(0)

        # critics
        critic1 = Critic(n_obs).cuda(0)
        critic2 = Critic(n_obs).cuda(0)
        critic3 = Critic(n_obs).cuda(0)

        # actor optimizers
        actor_optim1 = torch.optim.Adam(actor1.parameters(), IPPO_CONFIG["LR_Actor"])
        actor_optim2 = torch.optim.Adam(actor2.parameters(), IPPO_CONFIG["LR_Actor"])
        actor_optim3 = torch.optim.Adam(actor3.parameters(), IPPO_CONFIG["LR_Actor"])

        # critic optimizers
        critics_optim1 = torch.optim.Adam(critic1.parameters(), IPPO_CONFIG["LR_Critic"])
        critics_optim2 = torch.optim.Adam(critic2.parameters(), IPPO_CONFIG["LR_Critic"])
        critics_optim3 = torch.optim.Adam(critic3.parameters(), IPPO_CONFIG["LR_Critic"])

        # replay buffer
        replay_buffer1 = ReplayBuffer(IPPO_CONFIG["BUFFER_CAPACITY"])
        replay_buffer2 = ReplayBuffer(IPPO_CONFIG["BUFFER_CAPACITY"])
        replay_buffer3 = ReplayBuffer(IPPO_CONFIG["BUFFER_CAPACITY"])

        self.ep_i = 0

        self.replay_buffers = [replay_buffer1, replay_buffer2, replay_buffer3]
        self.actors = [actor1, actor2, actor3]
        self.critics = [critic1, critic2, critic3]
        self.actor_optims = [actor_optim1, actor_optim2, actor_optim3]
        self.critics_optims = [critics_optim1, critics_optim2, critics_optim3]

        self.actors_old = []
        self.critic_loss_sum = torch.tensor(0.).cuda()
        self.actor_loss_sum = torch.tensor(0.).cuda()

        self.loss_fn = nn.MSELoss()

    # add trajectory to the replay buffer
    def update_buffer(self, obs, actions, obs_, rewards, dones):
        for i in range(IPPO_CONFIG["N_AGENT"]):
            self.replay_buffers[i].push(obs[i], actions[i], obs_[i], rewards[i], dones[i])

    def train(self, ep_i):

        if ep_i != self.ep_i:  # train the agent at the end of an episode
            self.ep_i = ep_i
            self.critic_loss_sum = torch.tensor(0.).cuda()
            self.actor_loss_sum = torch.tensor(0.).cuda()

            for i in range(IPPO_CONFIG["N_AGENT"]):
                ob, action, ob_, reward, _ = self.replay_buffers[i].sample()
                ob = torch.Tensor(np.array(ob)).cuda(0)
                ob_ = torch.Tensor(np.array(ob_)).cuda(0)  # ob_ is the new state
                action = torch.Tensor(np.array(action)).cuda(0)
                reward = torch.Tensor(reward).cuda(0).view(action.shape[0], -1)
                self.replay_buffers[i].clean()

                # calculate target and advantage
                with torch.no_grad():
                    target = reward + IPPO_CONFIG["GAMMA"] * self.critics[i](ob_)
                    advantage = reward + target - self.critics[i](ob)

                self.actors_old = copy.deepcopy(self.actors)
                for _ in range(IPPO_CONFIG["N_UPDATE"]):
                    # update critic
                    self.critic_loss_sum += self.update_critic(i, ob, target)

                    # update actor
                    self.actor_loss_sum += self.update_actor(i, ob, action, advantage)

        return self.critic_loss_sum / IPPO_CONFIG["N_UPDATE"], self.actor_loss_sum / IPPO_CONFIG["N_UPDATE"]

    def update_critic(self, i, ob, target):
        v = self.critics[i](ob)
        loss = self.loss_fn(v, target)
        self.critics_optims[i].zero_grad()
        loss.backward()
        if IPPO_CONFIG['GRAD_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 10)
        self.critics_optims[i].step()
        return loss

    def update_actor(self, i, ob, action, advantage):
        a_prob = self.actors[i](ob)[:, action.argmax(dim=1)]
        a_prob_old = self.actors_old[i](ob)[:, action.argmax(dim=1)]
        ratio = a_prob / (a_prob_old + 1e-8)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - IPPO_CONFIG["EPSILON"], 1 + IPPO_CONFIG["EPSILON"]) * advantage
        loss = -torch.mean(torch.min(surr1, surr2))
        self.actor_optims[i].zero_grad()
        loss.backward()
        if IPPO_CONFIG['GRAD_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 10)
        self.actor_optims[i].step()
        return loss

    def choose_action(self, obs):
        if np.random.uniform() <= IPPO_CONFIG["EPS"]:
            actions = []
            for i in range(IPPO_CONFIG["N_AGENT"]):
                action = np.random.randint(0, IPPO_CONFIG["N_ACTION"])
                action = np.eye(IPPO_CONFIG["N_ACTION"])[action]
                actions.append(action)
        else:
            actions = self.act(obs)
        return actions

    def act(self, obs):
        torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1).cuda(0), requires_grad=False)
                     for i in range(IPPO_CONFIG["N_AGENT"])]
        actions = []
        for i in range(IPPO_CONFIG["N_AGENT"]):
            with torch.no_grad():
                action_continue = self.actors[i](torch_obs[i]).detach()
            actions.append(action_continue.detach().view(-1).cpu().numpy())
        return actions
