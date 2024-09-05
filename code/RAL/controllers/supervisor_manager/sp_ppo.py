## 基于sac实现的多车聚集任务
# 保存文件：models/actor_sac,critic_sac,critic_target_sac
import math


# from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv
from utilities import normalizeToRange
from controller import Supervisor
import numpy as np
import os

from sac_rl import *
from gym import spaces

import wandb

import torch

from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from models.util import Normalization, RewardScaling , PPOevaluate_policy, SACevaluate_policy
from models.replay_buffer import PPOReplayBuffer , SACReplayBuffer

from models.PPO_MODEL import PPO_continuous

def cal_dis(x,y):
    dis = np.linalg.norm([x,y])
    return dis

#定义robot：self.supervisor.getFrameDef('$robot_name')
#定义robot的emitter和receiver：self.supervisor.getDevice('$emitter_name')
# self.robot[i].getPosition() :获取第i个robot的x，y，z坐标
# self.robot[i].getVelocity():获取第i个robot的x，y，y方向的速度


class EpuckSupervisor:
    def __init__(self, num_robots=5):
        super().__init__()
        self.num_robots = num_robots
        self.num_cols = 2   #障碍物数量
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.steps = 0
        self.steps_limit = 1000
        self.communication = self.initialize_comms()
        self.action_space = spaces.Box(low = np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.ones(7),high = np.ones(7),dtype=np.float64)
        # 初始化reciver和emitter，存入communication

        # self.observation_space = 3
        self.obs_history = [ ]
        self.robot = [self.supervisor.getFromDef("e-puck" + str(i)) for i in range(self.num_robots)]
        #初始化robot
        self.col = [self.supervisor.getFromDef("col"+str(i)) for i in range(self.num_cols)]
        #初始化障碍物

        self.messageReceived = None
        self.episode_score = 0
        self.episode_score_list = []
        self.is_solved = False

        #初始位置：[-1.0,-0.7,0.7,1.0,0.7]
        #        [0.0,0.5,-0.5,0.0,0.5]
        self.targetx = [1.0,0.7,-0.7,-1.0,-0.7]
        self.targety = [0.0,0.5,0.5,0.0,-0.5]
        self.evaluate_reward_history = []


    def is_done(self):
        if self.steps >= self.steps_limit:
            return True
        else:
            return False

    def initialize_comms(self):
        communication = []
        for i in range(self.num_robots):
            emitter = self.supervisor.getDevice(f'emitter{i}')
            receiver = self.supervisor.getDevice(f'receiver{i}')

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            communication.append({
                'emitter': emitter,
                'receiver': receiver,
            })
        # communication : [{'emitter':**,'receiver':***},{'emitter':**,'receiver':***},{'emitter':**,'receiver':***}]

        return communication


    def step(self, action):
        if self.supervisor.step(self.timestep) == -1:
            exit()
        self.steps +=1
        self.handle_emitter(np.clip(action,-1.0,1.0))


        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info()
        )

    def handle_emitter(self, actions):
        for i, action in enumerate(actions):
            message = (",".join(map(str, action))).encode("utf-8")
            #获取第ige智能体的action：[,]
            # print(message)
            self.communication[i]['emitter'].send(message)
            #在第i个通道上发布message信息

    def handle_receiver(self):# 获取某个通道上发布的信息，存入messages中。
        messages = []
        for com in self.communication:
            receiver = com['receiver']
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getData().decode("utf8"))
                receiver.nextPacket()
            else:
                messages.append(None)
        # print(messages)
        return messages

    def get_observations(self):
        #self.robot[i].getPosition() :获取第i个robot的x，y，z坐标
        #self.robot[i].getVelocity():获取第i个robot的x，y，y方向的速度
        self.positions_x = np.array([normalizeToRange(self.robot[i].getPosition()[0], -1.97, 1.97, -2.0, 2.0)
                                for i in range(self.num_robots)])
        # 限制位置到-0.97
        self.positions_y = np.array([normalizeToRange(self.robot[i].getPosition()[1], -1.97, 1.97, -2.0, 2.0)
                                for i in range(self.num_robots)])
        #限制速度到0.15
        self.velocity_x = np.array([normalizeToRange(self.robot[i].getVelocity()[0], -0.15, 0.15, -1.0, 1.0)
                               for i in range(self.num_robots)])
        self.velocity_y = np.array([normalizeToRange(self.robot[i].getVelocity()[1], -0.15, 0.15, -1.0, 1.0)
                               for i in range(self.num_robots)])
        # print(self.distance)

        self.col_x = np.array([normalizeToRange(self.col[i].getField("translation").getSFVec3f()[0], -1.97, 1.97, -2.0, 2.0)
                                for i in range(self.num_cols)])
        # 限制位置到-1.97
        self.col_y = np.array([normalizeToRange(self.col[i].getField("translation").getSFVec3f()[1], -1.97, 1.97, -2.0, 2.0)
                                for i in range(self.num_cols)])



        self.messageReceived = self.handle_receiver()
        self.observations = np.empty((self.num_robots, self.observation_space.shape[0]), float)

        for i in range(self.num_robots):
            # a = np.append(positions_x[i]-positions_x,positions_y[i]-positions_y)
            # print('a',a)
            # self.observations[i] = np.append(a,[self.targetx-positions_x[i],self.targety-positions_y[i],self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
            #                                  if self.robot[i].getField("rotation").getSFRotation()[2] >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi)])
            # self.observations[i] = np.append([self.positions_x[i]-self.positions_x,self.positions_y[i]-self.positions_y],[self.targetx[i]-self.positions_x[i],
            #                                  self.targety[i]-self.positions_y[i], self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
            #                                  if self.robot[i].getField("rotation").getSFRotation()[2]
            #                                     >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi),self.velocity_x[i],self.velocity_y[i]
            #                                   ])
            self.observations[i] = np.append([self.targetx[i]-self.positions_x[i],
                                             self.targety[i]-self.positions_y[i]], [self.positions_x[i],self.positions_y[i],self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
                                             if self.robot[i].getField("rotation").getSFRotation()[2]
                                                >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi),self.velocity_x[i],self.velocity_y[i]
                                              ])
        self.obs_history.append(self.observations)
        return self.observations


        # for i in range(self.num_robots):
        #     for k in range(8):#将距离传感器参数模糊化，大于0.6表示有障碍物，值变为-1，小于0.6表示没有障碍物，值变为1
        #         if ds_value[i][k] >= 0.6:
        #             ds_value[i][k] = 1
        #         else:
        #             ds_value[i][k] = -1
        #     self.observations[i] = np.append(ds_value[i],[self.targetx-positions_x[i],
        #                                      self.targety- positions_y[i], self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
        #                                      if self.robot[i].getField("rotation").getSFRotation()[2] >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi),
        #                                       ])
        # self.obs_history.append(self.observations)
        # return self.observations




    def get_reward(self, action=None):
        """

        :param action:
        :type action:
        :return:
        """
        # self.ds : 距离越大，ds越小
        rewards = np.empty((self.num_robots,1),float)
        for i in range(self.num_robots):
            # dis_epu = []
            # for k in range(self.num_robots):
            #     dis_epu = cal_dis(self.observations[i][k],self.observations[i][k+self.num_robots])
            #     if dis_epu<=0.2 and dis_epu!=0.0:
            #         rewards[i] -= 0.1 # 自己定义的过于靠近的负向奖励

            # dis_goal = cal_dis(self.observations[i][10],self.observations[i][11])
            # if dis_goal < 0.2:
            #     rewards[i] += 0.2

            ds_current = cal_dis(self.observations[i][0],self.observations[i][1])
            ds_pre = cal_dis(self.obs_history[self.steps-2][i][0],self.obs_history[self.steps-2][i][1])
            rewards[i] = ds_pre - ds_current
        return rewards



    def get_default_observation(self):
        observation = []
        for _ in range(self.num_robots):
            robot_obs = [0.0 for _ in range(self.observation_space.shape[0])]
            observation.append(robot_obs)
        return observation

    def get_info(self):
        pass



    def reset(self):
        self.steps = 0
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        # px = 0.0016 * (np.random.choice(500, 5, replace=False) - 500)
        # py = 0.0016 * (np.random.choice(500, 5, replace=False) - 500)
        # for i in range(self.num_robots):
        #     self.robot[i].getField('translation').setSFVec3f([px[i],py[i],0])
        #     self.robot[i].getField('rotation').setSFRotation([0.0,0.0,1.0,0.0])
        # self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        # 暂时不需要随机初始化位置
        for i in range(self.num_robots):
            self.communication[i]['receiver'].disable()
            self.communication[i]['receiver'].enable(self.timestep)
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_default_observation()

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)



if __name__ =='__main__':
    # wandb.init(project='webots_sac',
    #            name='juji')



    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")


    args = parser.parse_args()

    create_path("./models/saved/ppo_all_map/")
    create_path("./exports_ppo_all_map/")
    env_name= 'juji'
    env=EpuckSupervisor()
    env_evaluate = env
    number = 1

    number2=1
    seed = 0
    alg='PPO'
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])

    args.max_episode_steps = env.steps_limit  # Maximum number of steps per episode
    total_steps = 0
    evaluate_num = 0
    evaluate_rewards = []
    std=[]
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))




    replay_buffer = PPOReplayBuffer(args.batch_size,args.state_dim,args.action_dim)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(
        log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        a = np.empty((env.num_robots, 2), float)
        a_logprob=np.empty((env.num_robots, 2), float)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1

            for n in range(env.num_robots):
                a[n],a_logprob[n] = agent.choose_action(s[n])
                s_, r, done, _ = env.step(a)


            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）

            for j in range(env.num_robots):

                replay_buffer.store(s[j], a[j], a_logprob[j],r[j], s_[j], dw,done)  # Store the transition


            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update

            if replay_buffer.count >= args.batch_size:
                agent.update(replay_buffer, total_steps)
                # std.append(torch.exp(agent.actor.log_std))
                # np.save('./data_train/enp_PPO_env_{}_number_{}.npy'.format(env_name, number),
                #         np.array(std))
                #
                replay_buffer.count = 0




            if (total_steps + 1) % args.evaluate_freq == 0:
                print(total_steps)
                print('eva')
                evaluate_num += 1
                evaluate_reward = PPOevaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

                if evaluate_num % 10 == 0:
                    np.save('./data_train/PPO_env_{}_number_{}.npy'.format(env_name, number),
                            np.array(evaluate_rewards))
