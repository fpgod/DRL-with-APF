## 基于sac实现的多车聚集任务
# 保存文件：models/actor_sac,critic_sac,critic_target_sac
# 观测是周围0.4内智能体得平均，到达目标点得奖励0.05 最终奖励大约为2219，保存得位置为C:/Users/fp/Downloads/all_obs/controllers/models/5
import math
import random

import torch
# from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv
from supervisor_manager.utilities import normalizeToRange
from controller import Supervisor
import numpy as np
import os
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
        self.observation_space = spaces.Box(low = -np.ones(7),high = np.ones(7),dtype=np.float64)
        # 初始化reciver和emitter，存入communication

        # self.observation_space = 3
        self.obs_history = [ ]
        self.dis_epu_history = [ ]
        self.selfless = [True for i in range(self.num_robots)]
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
        self.rot = np.array([self.robot[i].getField("rotation").getSFRotation()[3] % (2 * np.pi)
                                              if self.robot[i].getField("rotation").getSFRotation()[2]
                                                 > 0 else (-self.robot[i].getField("rotation").getSFRotation()[
                                                  3]) % (2 * np.pi) for i in range(self.num_robots)])
        # 限制速度到0.15
        self.velocity_x = np.array([normalizeToRange(self.robot[i].getVelocity()[0], -0.15, 0.15, -1.0, 1.0)
                                    for i in range(self.num_robots)])
        self.velocity_y = np.array([normalizeToRange(self.robot[i].getVelocity()[1], -0.15, 0.15, -1.0, 1.0)
                                    for i in range(self.num_robots)])
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
            if self.dis_col[i].min() > 0.4:
                c_temp = 0.4 #c[index_col]
                d_temp = 0.4 #d[index_col]
            else:
                c_temp = c[index_col]
                d_temp = d[index_col]
            if dis_epu_temp.min() > 0.4:
                # a_temp = np.mean(a)
                # b_temp = np.mean(b)
                a_temp = 0.4 #a[index]
                b_temp = 0.4 #b[index]
                self.selfless[i] = False
                v_x_temp = 0.0
                v_y_temp = 0.0
            else:
                a_temp_list = [ ]
                b_temp_list = [ ]
                t_temp = []
                for t in range(self.num_robots):
                    if 0.0<self.dis_epu[i][t] <=0.4:
                        # if self.dis_epu[i][t] >= self.dis_epu_history[-1][i][t]:
                        #     a_temp_list = a_temp_list
                        #     b_temp_list = b_temp_list
                        # else:
                        #     a_temp_list.append(a[t])
                        #     b_temp_list.append(b[t])
                        #     t_temp.append(t)
                        a_temp_list.append(a[t])
                        b_temp_list.append(b[t])
                        t_temp.append(t)
                    else:
                        a_temp_list = a_temp_list
                        b_temp_list = b_temp_list
                # a_temp = np.array(a_temp_list).mean()
                # b_temp = np.array(b_temp_list).mean()
                a_temp = a[index]
                b_temp = b[index]
                v_x_temp = self.velocity_x[index]
                v_y_temp = self.velocity_y[index]


                # dis_temp = [np.linalg.norm([a_temp_list[i],b_temp_list[i]]) for i in range(len(a_temp_list))]
                # dis_temp_temp = np.copy(dis_temp)
                # if len(dis_temp_temp):
                #     index_dis = np.argwhere(dis_temp_temp == dis_temp_temp.min())[0][0]
                #     index_min_epu = t_temp[index_dis]
                #     a_temp = a[index_min_epu]
                #     b_temp = b[index_min_epu]
                #     if self.selfless[index_min_epu]:
                #         self.selfless[i] =False
                #     else:
                #         self.selfless[i] = True
                # else:
                #     self.selfless[i] =False
                #     a_temp = 0.2
                #     b_temp = 0.2


            self.observations[i] = np.hstack([a_temp, b_temp, delta_x, delta_y,c_temp,d_temp,
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
        self.dis_epu_history.append(self.dis_epu)
        del self.dis_epu_history[:-2]
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
                rewards[i] += 0.05
            for k in range(self.num_robots):
                if 0.08 < self.dis_epu[i][k] < 0.12:
                    rewards[i] -= 0.15
                elif 0.0 < self.dis_epu[i][k] <= 0.08:
                    rewards[i] -= 0.35  #0.35
            # for k in range(self.num_robots):
            #     if 0.0 < self.dis_epu[i][k] <= 0.08:
            #         rewards[i] -= 0.10
            for j in range(self.num_cols):
                if 0.15 < self.dis_col[i][j] <= 0.19 :
                    rewards[i] -= 0.15
                elif 0.0 < self.dis_col[i][j] <= 0.15:
                    rewards[i] -= 0.35 # 0.35
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

    def apf(self):
        self.Kr = 0.5
        self.Ka = 5.0
        a_temp = np.empty((env.num_robots, 2), float)
        do = 0.2
        for i in range(self.num_robots):
            self.xGradRepPotFieldTemp = 0.0
            self.yGradRepPotFieldTemp = 0.0
            for k in range(self.num_cols):
                if self.dis_col[i][k] <=do:
                    Gx1 = (self.Kr * ((1 / self.dis_col[i][k]) - (1 / do)) * (1 /self.dis_col[i][k] ** 3) * (self.dis_goal[i]) * (
                                self.positions_x[i] - self.col_x[k]))
                    Gx2 = (0.5 * self.Kr * (((1 / self.dis_col[i][k]) - (1 / do)) ** 2) * (1 /self.dis_goal[i]) * (
                                self.positions_x[i] - self.targetx[i]))
                    Gy1 = (self.Kr * ((1 / self.dis_col[i][k]) - (1 / do)) * (1 / self.dis_col[i][k] ** 3) * (self.dis_goal[i]) * (
                                self.positions_y[i] - self.col_y[k]))
                    Gy2 = (0.5 * self.Kr * (((1 /self.dis_col[i][k]) - (1 / do)) ** 2) * (1 /self.dis_goal[i]) * (
                                self.positions_y[i] - self.targety[i]))
                    self.xGradRepPotFieldTemp = (Gx1 - Gx2) + self.xGradRepPotFieldTemp
                    self.yGradRepPotFieldTemp = (Gy1 - Gy2) + self.yGradRepPotFieldTemp
            self.xGradAttPotField = - self.Ka * 2 * (self.dis_goal[i])*(self.positions_x[i]-self.targetx[i])
            self.yGradAttPotField = - self.Ka * 2 * (self.dis_goal[i]) * (self.positions_y[i] - self.targety[i])
            for j in range(self.num_robots):
                if 0<self.dis_epu[i][j] <=do:
                    Gx3 = (self.Kr * ((1 / self.dis_epu[i][j]) - (1 / do)) * (1 /self.dis_epu[i][j] ** 3) * (self.dis_goal[i]) * (
                                self.positions_x[i] - self.positions_x[j]))
                    Gx4 = (0.5 * self.Kr * (((1 / self.dis_epu[i][j]) - (1 / do)) ** 2) * (1 /self.dis_goal[i]) * (
                                self.positions_x[i] - self.targetx[i]))
                    Gy3 = (self.Kr * ((1 / self.dis_epu[i][j]) - (1 / do)) * (1 / self.dis_epu[i][j] ** 3) * (self.dis_goal[i]) * (
                                self.positions_y[i] - self.positions_y[j]))
                    Gy4 = (0.5 * self.Kr * (((1 /self.dis_epu[i][j]) - (1 / do)) ** 2) * (1 /self.dis_goal[i]) * (
                                self.positions_y[i] - self.targety[j]))
                    self.xGradRepPotFieldTemp = (Gx3 - Gx4) + self.xGradRepPotFieldTemp
                    self.yGradRepPotFieldTemp = (Gy3 - Gy4) + self.yGradRepPotFieldTemp
            xGradPotField = self.xGradRepPotFieldTemp + self.xGradAttPotField
            yGradPotField = self.yGradRepPotFieldTemp + self.yGradAttPotField
            field_angle_temp = math.atan2(xGradPotField,yGradPotField)
            self.heading_angle = 1.57 - field_angle_temp
            w_temp = (self.heading_angle - self.rot[i])
            w_temp = w_temp + 0.5 * np.random.rand()
            if w_temp > 3.14:
                w_temp = w_temp - 6.28
            if w_temp < -3.14:
                w_temp = w_temp + 6.28
            # w_temp = (a- self.heading_angle)
            if w_temp <= 2.0 and w_temp >= -2.0:
                self.w = 1.5 * w_temp
            elif w_temp <= 3.14 and w_temp >= -3.14:
                self.w = 0.9 * w_temp
            else:
                self.w = 0.45 * w_temp
            a_temp[i][0] = (2 * 0.07536 - 0.0502 * self.w) / (2 * 0.0205)
            a_temp[i][1] = (2 * 0.07536 + 0.0502 * self.w) / (2 * 0.0205)
        return a_temp

    def reset(self):
        self.steps = 0
        col_radius = 0.1
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))

        #完全隨機障礙物位置
        # obstacles = []
        # while len(obstacles) < self.num_cols:
        #     x = random.uniform(-0.8 + col_radius, 0.8 - col_radius)
        #     y = random.uniform(-0.8 + col_radius, 0.8 - col_radius)
        #     new_obstacle = [x, y]
        #     overlap = False
        #     for obstacle in obstacles:
        #         if ((new_obstacle[0] - obstacle[0]) ** 2 + (new_obstacle[1] - obstacle[1]) ** 2) < (
        #                 2 * col_radius) ** 2:
        #             overlap = True
        #             break
        #     if not overlap:
        #         obstacles.append(new_obstacle)
        # for k in range(self.num_cols):
        #     self.col[k].getField('translation').setSFVec3f([obstacles[k][0],obstacles[k][1],0.15])
        #

        map_number = np.random.randint(0, 4)
        map_obstacles = [[[-0.5,0.5],[0.5,0.5],[0.0,0.0],[-0.5,-0.5],[0.5,-0.5]],\
                        [[-0.6,0.6],[0.6,0.6],[0.0,0.0],[-0.6,-0.6],[0.6,-0.6]],\
                        [[-0.4,0.4],[0.4,0.4],[0.0,0.0],[-0.4,-0.4],[0.4,-0.4]],\
                        [[-0.6,0.6],[0.4,0.4],[0.0,0.0],[-0.6,-0.6],[0.4,-0.4]],\
                        [[-0.4,0.4],[0.6,0.6],[0.0,0.0],[-0.4,-0.4],[0.6,-0.6]]]
        # for i in range(self.num_robots):
        #     self.robot[i].getField('translation').setSFVec3f([px[i],py[i],0])
        #     self.robot[i].getField('rotation').setSFRotation([0.0,0.0,1.0,0.0])
        #
        # self.supervisor.step(int(self.supervisor.getBasicTimeStep()))


        for k in range(self.num_cols):
            self.col[k].getField('translation').setSFVec3f([map_obstacles[map_number][k][0],map_obstacles[map_number][k][1],0.15])

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

    max_train_steps = 1e6  # Maximum number of training steps  3e6
    random_steps = 25e3  # Take the random actions in the beginning for the better exploration 25e3
    evaluate_freq = 10000  # Evaluate the policy every 'evaluate_freq' steps 5000
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    eposide_rewards = []

    while total_steps < max_train_steps:
        evaluate_num += 1
        evaluate_reward = evaluate_policy(env_evaluate, agent)
        evaluate_rewards.append(evaluate_reward)
        print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_reward, global_step=total_steps)
                # Save the rewards
                #if evaluate_num % 10 == 0:
                    #np.save('D:/fp/safe_rl_icra/all_obs/controllers/icra_safe/result/eva_env_{}_number_{}.npy'.format(env_name, number), np.array(evaluate_rewards))
        total_steps += 1




