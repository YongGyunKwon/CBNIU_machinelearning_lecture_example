import numpy as np
import matplotlib.pyplot as plt
import time
import copy

class Environment:
    cliff = -3;
    road = -1;
    sink = -2;
    goal = 2
    goal_position = [2, 3]
    reward_list = [[road, road, road, road], [road, road, sink, road], [road, road, road, goal]]
    reward_list1 = [['road', 'road', 'road', 'road'], ['road', 'road', 'sink', 'road'],
                    ['road', 'road', 'road', 'goal']]

    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    def move(self, agent, action):
        done = False
        new_pos = agent.pos + agent.action[action]

        if self.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[
            1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0], observation[1]]
        return observation, reward, done


class Agent:
    action = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    select_action_pr = np.array([0.25, 0.25, 0.25, 0.25])

    def __init__(self, initial_position):
        self.pos = initial_position

    def set_pos(self, position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos


def action_value_function(env, agent, act, G, max_step, now_step):
    gamma = 0.9

    if env.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal
    if max_step == now_step:
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward
        return G
    else:
        pos1 = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward

        if done == True:
            if observation[0]<0 or observation[0]>=env.reward.shape[0] or observation[1]<0 or observation[1]>=env.reward.shape[1]:
                agent.set_pos(pos1)

        pos1 = agent.get_pos()

        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            next_v = action_value_function(env, agent, i, 0, max_step, now_step+1)
            G += agent.select_action_pr[i]*gamma*next_v
        return G






def state_value_function(env, agent, G, max_step, now_step):
    gamma = 0.85
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal
    if max_step == now_step:
        pos1 = agent.get_pos()
        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            observation, reward, done = env.move(agent, i)
            G += agent.select_action_pr[i] * reward
        return G
    else:
        pos1 = agent.get_pos()
        for i in range(len(agent.action)):
            observation, reward, done = env.move(agent, i)
            G += agent.select_action_pr[i] * reward
            if done == True:
                if observation[0] < 0 or observation[0] >= env.reward.shape[0] or observation[1] < 0 or observation[
                    1] >= env.reward.shape[1]:
                    agent.set_pos(pos1)

            next_v = state_value_function(env, agent, 0, max_step, now_step + 1)
            G += agent.select_action_pr[i] * gamma * next_v
            agent.set_pos(pos1)
        return G


def show_v_table(v_table, env):
    for i in range(env.reward.shape[0]):
        print('+-----------------' * env.reward.shape[1], end='')
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    print('          |', end='')
                if k == 1:
                    print(' {0:8.2f} |'.format(v_table[i, j]), end='')
                if k == 2:
                    print('          |', end='')
            print()
        print('+-----------------' * env.reward.shape[1], end='')
        print('+')


def policy_extraction(env, agent, v_table, optimal_policy):
    gamma = 0.9
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp = -1e+10
            for action in range(len(agent.action)):
                agent.set_pos([i, j])
                observation, reward, done = env.move(agent, action)
                if temp < reward + gamma * v_table[observation[0], observation[1]]:
                    optimal_policy[i, j] = action
                    temp = reward + gamma * v_table[observation[0], observation[1]]
    return optimal_policy


def show_policy(policy, env):
    for i in range(env.reward.shape[0]):
        print('+----------' * env.reward.shape[1], end='');
        print('+');
        print('|', end='')
        for j in range(env.reward.shape[1]):
            if env.reward_list1[i][j] != 'goal':
                if policy[i, j] == 0:
                    print('   ↑   |', end='')
                elif policy[i, j] == 1:
                    print('   →   |', end='')
                elif policy[i, j] == 2:
                    print('   ↓   |', end='')
                elif policy[i, j] == 3:
                    print('   ←   |', end='')
            else:
                print('   *   |', end='')
        print()







def policy_evaluation(env,agent,v_table,policy):
    while True:
        delta=0
        temp_v= copy.deepcopy(v_table)
        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                agent.set_pos([i,j])
                action=policy[i,j]
                observation, reward, done = env.move(agent,action)
                v_table[i,j]=reward+gamma*v_table[observation[0],observation[1]]
        delta=np.max([delta,np.max(np.abs(temp_v-v_table))])
        if delta < 0.000001:
            break
    return v_table,delta

def policy_improvement(env,agent,v_table,policy):
    policyStable=True
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            old_action=policy[i,j]
            temp_action = 0
            temp_value = -1e+10
            
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done= env.move(agent,action)
                if temp_value < reward+gamma*v_table[observation[0],observation[1]]:
                    temp_action=action
                    temp_value = reward + gamma*v_table[observation[0],observation[1]]
                if old_action !=temp_action:
                    policyStable=False
                policy[i,j]=temp_action
    return policy, policyStable

np.random.seed(0)
env=Environment()
initial_position=np.array([0,0])
agent=Agent(initial_position)
gamma=0.0

v_table=np.random.rand(env.reward.shape[0],env.reward.shape[1])
policy=np.random.randint(0,4,(env.reward.shape[0],env.reward.shape[1]))

max_iter_number=20000

for iter_number in range(max_iter_number):
    v_table,delta=policy_evaluation(env,agent,v_table,policy)
    policy, policyStable = policy_improvement(env,agent,v_table,policy)
    show_v_table(v_table,env)
    show_policy(policy,env)

    if policyStable==True:
        break
