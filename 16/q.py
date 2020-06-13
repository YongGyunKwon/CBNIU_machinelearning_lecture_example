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


def generate_episode_with_policy(env,agent,first_vist,policy):
    gamma=0.09
    episode=[ ]
    visit=np.zeros((env.reward.shape[0],env.reward.shape[1],len(agent.action)))

    i=0;j=0
    agent.set_pos([i,j])
    G=0; step=0; max_step=100

    for k in range(max_step):
        pos=agent.get_pos()

        action=np.random.choice(range(0,len(agent.action)), p=policy[pos[0],pos[1],:]) 
        observation, reward, done= env.move(agent,action)
        if first_vist:
            if visit[pos[0],pos[1],action]==0:
                G+=gamma**(step)*reward
                visit[pos[0],pos[1],action]=1
                step+=1
                episode.append((pos,action,reward))

        else:
            G+=gamma**(step)*reward
            step+=1
            episode.append((pos,action,reward))

        if done==True:
            break
    
    return i,j,G,episode


def show_q_table(q_table, env):
    for i in range(env.reward.shape[0]):
        print('+-------' * env.reward.shape[1], end='');
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    print('{0:10.2f}    '.format(q_table[i, j, 0]), end='')
                if k == 1:
                    print('{0:6.2f}  {1:6.2f} |'.format(q_table[i, j, 3], q_table[i, j, 1]), end='')
                if k == 2:
                    print('{0:10.2f}    '.format(q_table[i, j, 2]), end='')
            print()
    print('+----------' * env.reward.shape[1], end='');
    print('+')


def show_q_table_arrow(q_table, env):
    for i in range(env.reward.shape[0]):
        print('+----------' * env.reward.shape[1], end='');
        print('+')
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    if np.max(q[i, j, :]) == q[i, j, 0]:
                        print('   ↑   |', end='')
                    else:
                        print('       |', end='')
                if k == 1:
                    if np.max(q[i, j, :]) == q[i, j, 1] and np.max(q[i, j, :]) == q[i, j, 3]:
                        print('  ← →  |', end='')
                    elif np.max(q[i, j, :]) == q[i, j, 1]:
                        print('    →  |', end='')
                    elif np.max(q[i, j, :]) == q[i, j, 3]:
                        print('  ←    |', end='')
                    else:
                        print('       |', end='')
                if k == 2:
                    if np.max(q[i, j, :]) == q[i, j, 2]:
                        print('   ↓   |', end='')
                    else:
                        print('       |', end='')
            print()
    print('+----------' * env.reward.shape[1], end='');
    print('+')


def greedy(Q_table, agent, epsilon):
    pos=agent.get_pos()
    return np.argmax(Q_table[pos[0], pos[1], :])

def e_greedy(Q_table, agent, epilson):
    pos=agent.get_pos()
    greedy_action=np.argmax(Q_table[pos[0],pos[1], :])
    pr=np.zeros(len(agent.action))
    for i in range(len(agent.action)):
        if i == greedy_action:
            pr[i]=1-epilson+epilson/len(agent.action)
        
        else:
            pr[i]=epilson/len(agent.action)
    
    return np.random.choice(range(0,len(agent.action)),p=pr)

np.random.seed(0)
env=Environment()
initial_position=np.array([0,0])
agent=Agent(initial_position)
gamma=0.9
Q_table=np.random.rand(env.reward.shape[0],env.reward.shape[1],len(agent.action))
Q_table[2,3,:]=0
max_episode=10000; max_step=100
alpha=0.1; epsilon=0.8

for epi in range(max_episode):
    delta=0;i=0;j=0
    agent.set_pos([i,j]); temp=0

    for k in range(max_step):
        pos=agent.get_pos()
        action = e_greedy(Q_table,agent,epsilon)
        observation, reward, done= env.move(agent, action)
        next_action=greedy(Q_table,agent,epsilon)
        Q_table[pos[0],pos[1],action]+=alpha*(reward+gamma*Q_table[observation[0],observation[1],next_action]
            -Q_table[pos[0],pos[1],action]
        )

        action=next_action
        if done==True:
            break

optimal_policy=np.zeros((env.reward.shape[0],env.reward.shape[1]))

for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        optimal_policy[i,j]=np.argmax(Q_table[i,j,:])

print('Q-learning: Q(s,a)')
show_q_table(np.round(Q_table,2),env)
print('Q-learning 최적정책')
show_policy(optimal_policy,env)