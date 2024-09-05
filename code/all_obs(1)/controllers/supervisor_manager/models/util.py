import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)



def SACevaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        a = np.empty((env.num_robots, 2), float)
        while not done:
            for i in range(env.num_robots):
                a[i]= agent.choose_action(s[i], deterministic=True)  # We use the deterministic policy during the evaluating
                s_, r, done, _ = env.step(a)
                episode_reward += r[i]
                s = s_
        #calculate std
        evaluate_reward += episode_reward
    env.evaluate_reward_history.append(evaluate_reward / times)
    if evaluate_reward/times == max(env.evaluate_reward_history):
        agent.save_models()
    return int(evaluate_reward / times)


def PPOevaluate_policy(args, env, agent, state_norm):
    times = 3

    evaluate_reward = 0
    for _ in range(times):

        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        a = np.empty((env.num_robots, 2), float)
        while not done:
            for n in range(env.num_robots):
                a[n] = agent.evaluate(s[n])
                s_, r, done, _ = env.step(a)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times