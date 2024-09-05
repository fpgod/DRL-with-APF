## 基于sac实现的多车聚集任务
# 保存文件：models/actor_sac,critic_sac,critic_target_sac
import math

import torch
# from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv
from utilities import normalizeToRange
from controller import Supervisor
import numpy as np
import os
from models.networks import DDPG
from sac_rl import *
from gym import spaces

# import wandb

def cal_dis(x,y):
    dis = np.linalg.norm([x,y])
    return dis

#定义robot：self.supervisor.getFrameDef('$robot_name')
#定义robot的emitter和receiver：self.supervisor.getDevice('$emitter_name')
# self.robot[i].getPosition() :获取第i个robot的x，y，z坐标
# self.robot[i].getVelocity():获取第i个robot的x，y，y方向的速度


class EpuckSupervisor:
    def __init__(self, num_robots=10):
        super().__init__()
        self.num_robots = num_robots
        self.num_cols = 5   #障碍物数量
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.steps = 0
        self.steps_limit = 1000
        self.communication = self.initialize_comms()
        self.action_space = spaces.Box(low = np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.ones(33),high = np.ones(33),dtype=np.float64)
        # 初始化reciver和emitter，存入communication

        # self.observation_space = 3
        self.obs_history = [ ]
        self.robot = [self.supervisor.getFromDef("e-puck" + str(i)) for i in range(self.num_robots)]
        self.col = [self.supervisor.getFromDef("col"+str(i)) for i in range(self.num_cols)]

        #初始化robot

        self.messageReceived = None
        self.episode_score = 0
        self.episode_score_list = []
        self.is_solved = False
        self.targetx = [1.00,-1.0,1.00,0.00,-1.00,1.00,-1.00,0.00,-0.5,0.50]
        self.targety = [-1.00,-1.0,0.00,-1.00,0.00,1.00,1.00,1.00,-1.00,1.00]
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
        self.handle_emitter(np.clip(action,-1.0,1.0))
        if self.supervisor.step(self.timestep) == -1:
            exit()
        self.steps +=1

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
        self.positions_x = np.array([normalizeToRange(self.robot[i].getPosition()[0], -1.97, 1.97, -2.0, 2.0)
                                     for i in range(self.num_robots)])
        # 限制位置到-0.97
        self.positions_y = np.array([normalizeToRange(self.robot[i].getPosition()[1], -1.97, 1.97, -2.0, 2.0)
                                     for i in range(self.num_robots)])
        # 限制速度到0.15
        # self.velocity_x = np.array([normalizeToRange(self.robot[i].getVelocity()[0], -0.15, 0.15, -1.0, 1.0)
        #                             for i in range(self.num_robots)])
        # self.velocity_y = np.array([normalizeToRange(self.robot[i].getVelocity()[1], -0.15, 0.15, -1.0, 1.0)
        #                             for i in range(self.num_robots)])
        # print(self.distance)

        self.col_x = np.array(
            [normalizeToRange(self.col[i].getField("translation").getSFVec3f()[0], -1.97, 1.97, -2.0, 2.0)
             for i in range(self.num_cols)])
        # 限制位置到-1.97
        self.col_y = np.array(
            [normalizeToRange(self.col[i].getField("translation").getSFVec3f()[1], -1.97, 1.97, -2.0, 2.0)
             for i in range(self.num_cols)])
        self.messageReceived = self.handle_receiver()
        # ds_value = np.empty([self.num_robots, 8], float)
        # for i, message in enumerate(self.messageReceived):
        #     if message is not None:
        #         message = message.split(',')
        #         ds_value[i] = [message[j] for j in range(8)]
        #         for k in range(8):
        #             ds_value[i][k] =(ds_value[i][k]-60)/60
        #         self.ds = ds_value
        #     else:
        #         ds_value = np.zeros((self.num_robots, 8), float)
        self.observations = np.empty((self.num_robots, self.observation_space.shape[0]), float)
        self.dis_goal = [cal_dis(self.positions_x[i] - self.targetx[i], self.positions_y[i] - self.targety[i]) for i
                         in range(self.num_robots)]
        self.rot_goal = [math.atan2(self.targety[i] - self.positions_y[i], self.targetx[i] - self.positions_x[i])
                         for i in range(self.num_robots)]
        self.dis_epu = np.empty((self.num_robots, self.num_robots), float)
        self.rot_epu = np.empty((self.num_robots, self.num_robots), float)
        self.dis_col = np.empty((self.num_robots,self.num_cols),float)
        for i in range(self.num_robots):
            a = self.positions_x - self.positions_x[i]
            b = self.positions_y - self.positions_y[i]
            c = self.positions_x[i] - self.col_x
            d = self.positions_y[i] - self.col_y
            for k in range(self.num_robots):
                self.dis_epu[i][k] = cal_dis(a[k], b[k])
                self.rot_epu[i][k] = math.atan2(b[k], a[k])
            for j in range(self.num_cols):
                self.dis_col[i][j] = cal_dis(c[j],d[j])
            dis_col_temp = np.copy(self.dis_col[i])
            index_col = np.argwhere(dis_col_temp  == dis_col_temp .min())[0][0]
            dis_epu_temp = np.copy(self.dis_epu[i])
            dis_epu_temp[np.where(self.dis_epu[i] == 0)] = 999
            index = np.argwhere(dis_epu_temp == dis_epu_temp.min())[0][0]
            # a = np.append(positions_x[i]-positions_x,positions_y[i]-positions_y)
            # print('a',a)
            # self.observations[i] = np.append(a,[self.targetx-positions_x[i],self.targety-positions_y[i],self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
            #                                 if self.robot[i].getField("rotation").getSFRotation()[2] >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi)])
            # self.observations[i] = np.append([self.dis_epu[i],self.rot_epu[i]],[self.dis_goal[i],
            #                                  self.rot_goal[i], self.positions_x[i]-self.targetx[i],self.positions_y-self.targety[i],self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
            #                                  if self.robot[i].getField("rotation").getSFRotation()[2]
            #                                     >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi)
            #                                   ])
            delta_x = self.positions_x[i] - self.targetx[i]
            delta_y = self.positions_y[i] - self.targety[i]
            self.observations[i] = np.hstack([a, b, delta_x, delta_y,c,d,
                                              self.robot[i].getField("rotation").getSFRotation()[3] % (2 * np.pi)
                                              if self.robot[i].getField("rotation").getSFRotation()[2]
                                                 > 0 else (-self.robot[i].getField("rotation").getSFRotation()[
                                                  3]) % (2 * np.pi)
                                              ])
            # self.observations[i] = np.append([self.positions_x[i]],[self.positions_y[i],ds_value[i],self.dis_goal[i],
            #                      self.rot_goal[i],self.robot[i].getField("rotation").getSFRotation()[3]%(2*np.pi)
            #                      if self.robot[i].getField("rotation").getSFRotation()[2]
            #                         >0 else (-self.robot[i].getField("rotation").getSFRotation()[3])%(2*np.pi)
            #                       ])
        self.obs_history.append(self.observations)
        del self.obs_history[:-2]
        return self.observations

    def get_reward(self, action=None):
        """

        :param action:
        :type action:
        :return:
        """
        # self.ds : 距离越大，ds越小
        rewards = np.empty((self.num_robots, 1), float)
        for i in range(self.num_robots):
            # for k in range(self.num_robots):
            #     if self.observations[i][k] < 0.1 and self.observations[i][k]!=0:
            #         rewards[i] -= 0.1
            # 自己定义的过于靠近的负向奖励
            rewards[i] = 100 * (np.linalg.norm([self.obs_history[-2][i][2], self.obs_history[-2][i][3]]) - \
                                np.linalg.norm([self.obs_history[-1][i][2], self.obs_history[-1][i][3]]))
            # rewards[i] = - np.linalg.norm([self.positions_x[i]-self.targetx[i],self.positions_y[i]-self.targety[i]])
            if self.dis_goal[i]<0.03:
                rewards[i] += 0.3
            for k in range(self.num_robots):
                if 0.08 < self.dis_epu[i][k] < 0.12:
                    rewards[i] -= 0.15
                elif 0.0 < self.dis_epu[i][k] <= 0.08:
                    rewards[i] -= 0.35
            for j in range(self.num_cols):
                if 0.18 < self.dis_col[i][j] <= 0.21 :
                    rewards[i] -= 0.15
                elif 0.0 < self.dis_epu[i][j] <= 0.18:
                    rewards[i] -= 0.35
            # rewards[i] -= self.observations[i][10]
            # if(action[i][0]==0.0 and action[i][1]==0.0):
            #     rewards[i] = 0
            # else:
            #     rewards[i] -=0.05

            # dis_goal = cal_dis(self.observations[i][10],self.observations[i][11])
            # if dis_goal < 0.5:
            #     rewards[i] += 1.5- dis_goal

            # ds_current = cal_dis(self.observations[i][0], self.observations[i][1])
            # ds_pre = cal_dis(self.obs_history[self.steps - 2][i][0], self.obs_history[self.steps - 2][i][1])
            # rewards[i] = ds_pre - ds_current
        return rewards


    def get_default_observation(self):
        observation = []
        for _ in range(self.num_robots):
            robot_obs = [0.0 for _ in range(self.observation_space.shape[0])]
            observation.append(robot_obs)
        self.obs_history.append(observation)
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
        #
        # self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        for i in range(self.num_robots):
            self.communication[i]['receiver'].disable()
            self.communication[i]['receiver'].enable(self.timestep)
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_observations()

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)



if __name__ =='__main__':
    # wandb.init(project='nav_all_pos',
    #            name='1')
    env_name= 'nav'
    env=EpuckSupervisor()
    env_evaluate = env
    number = 1
    seed = 41
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = env.steps_limit  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))
    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    max_train_steps = 3e6  # Maximum number of training steps  3e6
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration 25e3
    evaluate_freq = 10000  # Evaluate the policy every 'evaluate_freq' steps 5000
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    eposide_rewards = []

    while total_steps < max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        a = np.empty((env.num_robots, 2), float)
        ep_e = 0
        eposide_reward = 0
        while not done:
            episode_steps += 1
            for n in range(env.num_robots):
                if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    a[n] = env.action_space.sample()
                else:
                    a[n]= agent.choose_action(s[n])
            s_, r, done, _ = env.step(a)
            eposide_reward += r
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done:
                dw = True
                eposide_rewards.append(eposide_reward)
                np.save('D:/fp/nav/10/col/all_obs/controllers/models/921/SAC_near_r_{}_number_{}.npy'.format(env_name, number),
                        np.array(eposide_rewards))
            else:
                dw = False
            for j in range(env.num_robots):
                replay_buffer.store(s[j], a[j], r[j], s_[j], dw)  # Store the transition
            s = s_

            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                print(total_steps)
                print('eva')
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('D:/fp/nav/10/col/all_obs/controllers/models/921/SAC_env_{}_number_{}.npy'.format(env_name, number), np.array(evaluate_rewards))
            total_steps += 1




