def state_value_fuction(env,agent,G,max_step,now_step):
gamma=0.85
if env.reward_list1[agent.pos[0]][agent.pos[1]]=='goal':
    return env.goal

if max_step==now_step:
    pos1=agent.get_pos()
