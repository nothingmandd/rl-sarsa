import gym
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns; sns.set()

def parse_args():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default='1000')
    parser.add_argument('--discount_factor', type=float, default='0.9')
    parser.add_argument('--learning_rate', type=float, default='0.5')
    
    return parser.parse_args()

def q_learning(env, episodes, discount_factor, learning_rate):
    
    # initialize 3D array 
    q_values = np.zeros(shape = (pos_num, vel_num, env.action_space.n))
    
    # assign random uniform values for each element
    for i in range(pos_num):
        for j in range(vel_num):
            for k in range(env.action_space.n):
                q_values[i,j,k] = np.random.uniform(low=-0.5, high=0.5)
    
    # randomly select an action
    past_action = env.action_space.sample()
    
    for i in range(episodes):
        
        # discretize the state and initialize necessary variables
        exact_s = env.reset()
        state = np.round((exact_s- env.observation_space.low)*np.array([10, 100])).astype(int)
        done = False
        timestep = 0
        
        while(done != True): 
            
            # render the last 10 episodes
            #if (i-episodes < 10):env.render()  
            
            # choose the action with max Q-value, observe variables after performing it
            action = np.argmax(q_values[state[0], state[1],:])
            next_exact_s, r, done, info = env.step(action)
            
            # discretize the new state
            next_state = np.round((next_exact_s- env.observation_space.low)*np.array([10, 100])).astype(int)
            
            # update the Q-value of state-action pair 
            q_values[state[0], state[1], action] += learning_rate*(r + 
                                                        discount_factor*q_values[next_state[0], next_state[1], past_action] 
                                                                   - q_values[state[0], state[1], action]) 
            # update necessary variables
            state = next_state
            timestep += 1
            past_action = action
            
        if (i == 0 | i % 50 == 0): print("Episode %d finished after %d timesteps." % (i + 1,timestep))
    env.close()
    return q_values

def main():
    args = parse_args()
    
    np.random.seed(99)
    env = gym.make("MountainCar-v0")

    # get the size of the discretized state space 
    state_space = np.array([env.observation_space.high - env.observation_space.low])
    pos_num = int(state_space[0,0]*10) + 1
    vel_num = int(state_space[0,1]*100) + 1
    q_array = q_learning(env, args.episodes, args.discount_factor, args.learning_rate)

    # initialize and update the state-value function
    state_values = np.zeros(shape = (pos_num, vel_num))
    for i in range(pos_num):
        for j in range(vel_num):
            for k in range(env.action_space.n):
                state_values[i,j] = np.amax(q_array[i,j,:])

    # heat map of the state-value function
    ax = sns.heatmap(state_values)

if __name__ == "__main__":
    main()