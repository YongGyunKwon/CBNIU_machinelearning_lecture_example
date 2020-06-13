import numpy as np 
import matplotlib.pyplot as plt 
import time 

class Environment:
    cliff=-3; road=-1; sink=-2; goal = 2
    goal_position=[2,3];
    reward_list=[[road,road,road,road],[road,road,sink,road],[road,road,road,goal]]
    reward_list1=[['road','road','road','road'],['road','road','sink','road'],['road','road','road','goal']]

    def __init__(self):
            self.reward=np.asarray(self.reward_list)


    def move(self,agent,action):
        done=False
        new_pos=agent.pos+agent.action[action]

        if self.reward_list1[agent.pos[0]][agent.pos[1]]=='goal':
            reward=self.goal
            observation=agent.set_pos(agent.pos)
            done=True
        elif new_pos[0]<0 or new_pos[0]>=self.reward.shape[0] or new_pos[1]<0 or new_pos[1]>=self.reward.shape[1]:
            reward=self.cliff
            observation=agent.set_pos(agent.pos)
            done=True
        else:
            observation=agent.set_pos(new_pos)
            reward=self.reward[observation[0], observation[1]]
        
        return observation, reward, done 

class Agent:
    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    select_action_pr = np.array([0.25,0.25,0.25,0.25])

    def __init__(self,initial_position):
        self.pos=initial_position

    def set_pos(self,position):
        self.pos=position
        return self.pos

    def get_pos(self):
        return self.pos

def state_value_function(env,agent,G,max_step,now_step):
    gamma=0.85

    if env.reward_list1[agent.pos[0]][agent.pos[1]]=='goal':
        return env.goal

    if max_step==now_step:
        pos1=agent.get_pos()
        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            observation, reward, done= env.move(agent,i)
            G+=agent.select_action_pr[i]*reward


        return G
    
    else:
        pos1=agent.get_pos()
        for i in range(len(agent.action)):
            observation, reward, done = env.move(agent,i)
            G+=agent.select_action_pr[i]*reward
            if done == True:
                if observation[0]<0 or observation[0]>=env.reward.shape[0] or observation[1]<0 or observation[1]>=env.reward.shape[1]:
                    agent.set_pos(pos1)
            
            next_v=state_value_function(env,agent,0,max_step,now_step+1)
            G+=agent.select_action_pr[i]*gamma*next_v
            agent.set_pos(pos1)
        
        return G
    

def show_v_table(v_table,env):
    for i in range(env.reward.shape[0]):
        print('+--------------'*env.reward.shape[1], end=")
        print('+')
        for k in range(3):
            print('|',end=")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print('          |',end=")
                    if k==1:
                        print('   {0:8.2f}     |'.format(v_table[i,j]),end=")

                    if k==2:
                        print('        |',end=")
                print()
        print('+----------'*env.reward.shape[1],end=")
        print('+')


def policy_extraction(env,agent,v_table,optimal_policy):
    gamma=0.9
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp= -1e+10
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent,action)
                if temp<reward+gamma *v_table[observation[0],observation[1]]:
                    optimal_policy[i,j]=action
                    temp=reward+gamma*v_table[observation[0],observation[1]]
    return optimal_policy

def show_policy(policy,env):
    for i in range(env.reward.shape[0]):
        print('+----------'*env.reward.shape[1], end=" ")
        print('+')
        print('|',end=" ")
        for j in range(env.reward.shape[1]):
            if env.reward_list1[i][j]!='goal':
                if policy[i][j]==0:
                    print('     ↑     |',end=")
                elif policy[i][j]== 1:
                    print('     →     |',end=")
                elif policy[i][j]== 2:
                    print('     ↓     |',end=")
                elif policy[i][j]== 3:
                    print('     ←     |',end=")
                    
            else:
                print('     *     |',end=" ")

        print()
    print('+----------'*env.reward.shape[1], end="))
    print('+')


env=Environment()
initial_position=np.array([0,0])
agent=Agent(initial_position)
max_step_number=11
time_len=[ ]

for max_step in range(max_step_number):
    v_table=np.zeros((env.reward.shape[0], env.reward.shape[1]))
    start_time=time.time()
    for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                agent.set_pos([i,j])
                v_table[i,j] = state_value_function(env,agent,0,max_step,0)
    
    time_len.append(time.time()-start_time)
    print('\n 단계 수 = {} 실행시간 = {}'.format(max_step,np.round(time.time()-start_time,2)))
    show_v_table(np.round(v_table,2),env)

print('Reward')
show_v_table(np.array(env.reward_list),env)


policy=np.zeros((env.reward.shape[0],env.reward.shape[1]))
policy=policy_extraction(env,agent,v_table,policy)
show_policy(policy,env)
plt.plot(time_len,'o-k')
plt.xlabel('max steps')
plt.ylabel('time(s')
plt.show()