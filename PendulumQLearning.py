import gym
import numpy as np
import random
import matplotlib.pyplot as plt 

env = gym.make("Pendulum-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20_000
NUM_BINS = 40
ACTION_VAR = 1000 
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [NUM_BINS] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high+1 - env.observation_space.low) / DISCRETE_OS_SIZE
#print('discrete_os_win_size:', discrete_os_win_size)

action_space = np.linspace(int(env.action_space.low), int(env.action_space.high), ACTION_VAR)

epsilon = 0.2
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [len(action_space)]))
q_table = np.load("qtables/result-qtable.npy") #load q_table from previous  run

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    episode_reward = 0
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    while not done:
        # action from Q table
        if np.random.random() > epsilon:
            action = [action_space[np.argmax(q_table[discrete_state])]]
            action_index = np.argmax(q_table[discrete_state])
        # random action     
        else:
            action = [random.choice(action_space)]
            action_index = np.where(action_space == action[0])

        new_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        
        new_discrete_state = get_discrete_state(new_state)
        
        if episode % SHOW_EVERY == 0:
            env.render()
            
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action_index,)]
            new_q = current_q*(1 - LEARNING_RATE) + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action_index, )] = new_q
            
            q_table[discrete_state + (action_index, )] = new_q
            
        elif reward >= -0.1:
            q_table[discrete_state + (action_index, )] = 0    
        
        discrete_state = new_discrete_state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value  
        
    ep_rewards.append(episode_reward) # add episode reward to the list ep_rewards
    
    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
        print(f'Episode: {episode}, average reward: {average_reward}, min reward: {min(ep_rewards[-SHOW_EVERY:])}, max reward: {max(ep_rewards[-SHOW_EVERY:])}')
    
np.save("qtables/result-qtable.npy", q_table)

env.close()    

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.title(f"Reward over {EPISODES} episodes")
plt.legend(loc=4)
plt.grid(True)
plt.show()