import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action,name):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)
        chkpt_dir = 'D:/fp/nav/10/col/all_obs/controllers/models/921'
        self.checkpoint_file = os.path.join(chkpt_dir ,name+"_sac10_1")

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        # wandb.log({'std0': std[0][0]})
        # wandb.log({'std1': std[0][1]})

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi

    def save_checkpoint(self):
        print("...saving checkpoint....")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width,name):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)
        chkpt_dir = 'D:/fp/nav/10/col/all_obs/controllers/models/921'
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_sac10_1")

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def save_checkpoint(self):
        print("...saving checkpoint....")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(torch.load(self.checkpoint_file))

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        self.load_model = False
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1,requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2
        if self.load_model:
            self.load_models(state_dim,action_dim,max_action)
            self.load_model = True
        else:
            self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action,name='actor')
            self.critic = Critic(state_dim, action_dim, self.hidden_width,name='critic')
            # self.critic_target = copy.deepcopy(self.critic)
            self.critic_target = Critic(state_dim, action_dim, self.hidden_width, name='critic_target')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _= self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.cpu().numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch
        batch_s = batch_s
        batch_a = batch_a
        batch_r = batch_r
        batch_dw = batch_dw
        batch_s_ = batch_s_
        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            # print(self.alpha.is_cuda,batch_r.is_cuda,batch_dw.is_cuda,target_Q1.is_cuda,target_Q2.is_cuda,log_pi_.is_cuda)
            target_Q = batch_r + 0.99 * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha* log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self,state_dim,action_dim,max_action):
        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action, name='actor')
        self.critic = Critic(state_dim, action_dim, self.hidden_width, name='critic')
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_target = Critic(state_dim, action_dim, self.hidden_width, name='critic_target')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

def evaluate_policy(env, agent):
    times = 1  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        a = np.empty((env.num_robots, 2), float)
        while not done:
            for i in range(env.num_robots):
                a[i]= agent.choose_action(s[i], deterministic=True)  # We use the deterministic policy during the evaluating
                # if(s[i][10]<0.1):
                #     a[i] = [0.0,0.0]
            s_, r, done, _ = env.step(a)
            for k in range(env.num_robots):
                episode_reward += r[k]
            s = s_
        #calculate std
        evaluate_reward += episode_reward
    env.evaluate_reward_history.append(evaluate_reward / times)
    if evaluate_reward/times == max(env.evaluate_reward_history):
        agent.save_models()
    return int(evaluate_reward / times)




