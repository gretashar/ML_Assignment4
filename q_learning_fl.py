import numpy as np
import time
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import time
import sys
import matplotlib.image as mpimg
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import glob
import math
import random as rd


def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state,_ = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # select a greedy action
            next_state, rew, done, __,_ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew 
            if done:
                state,_ = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!'%(np.mean(tot_rew), num_episodes))

    return np.mean(tot_rew)

    

            
def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005, greedy_=False):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
    local_rewards=[]
    average_reward=[]
    staggered=[]
    sz_arr=[]
    
    for ep in range(num_episodes):
        state,_ = env.reset()
        done = False
        tot_rew = 0
        
        
        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            if greedy_:
                action = greedy(Q, state)
            else:
                action = eps_greedy(Q, state, eps)

            next_state, rew, done, __,_ = env.step(action) # Take one step in the environment

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        local_rewards.append(tot_rew)
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            # print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)




           
    return Q, local_rewards

def run(length=20, greedy_=False):
    epsilon = [0.05, 0.1, 0.125, 0.15, 0.25, 0.375, 0.5, 0.75, 0.90]
    time_fl=[]
    sz_arr_all=[]
    average_reward_all=[]
    global_rewards=[]
    staggered=[]
    print(length)
    for eps in epsilon:
        np.random.seed(0)
        this_map = generate_random_map(size=length)
        env = gym.make("FrozenLake-v1", desc=this_map)
        # env = gym.make("FrozenLake-v1")
        env = env.unwrapped
        desc = env.unwrapped.desc
        env.render()  # check environment
        num_episodes=10000
        start = time.time()
        Q_qlearning,local_rewards = Q_learning(env, lr=.1, num_episodes=10000, eps=eps, gamma=0.95, eps_decay=0.001, greedy_=greedy_)
        end = time.time()
        # sz_arr_all.append(sz_arr)
        # average_reward_all.append(average_reward)
        time_fl.append(end-start)
        global_rewards.append(local_rewards)

        average_reward_all.append(averages)
        
    print(sz_arr_all[0])
    if greedy_:
        plt.plot(range(0, len(global_rewards[0]), sz_arr_all[0]), average_reward_all[0],label='epsilon=0')
    else:
        
        for i, eps in enumerate(epsilon):    
            plt.plot(range(0, len(global_rewards[i]), sz_arr_all[i]), average_reward_all[i],label='epsilon='+str(eps))

    if greedy_:
        title=f"Frozen Lake {length}x{length} Q-Learning Greedy Rewards"
    else:
        title=f"Frozen Lake {length}x{length} Q-Learning Epsilon Greedy Rewards"
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=3, fancybox=True, shadow=True)
    plt.legend(loc=2, prop={'size': 6})
    plt.ylabel('Reward')
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()

    if greedy_:
        title=f"Frozen Lake {length}x{length} Q-Learning Greedy Time"
    else:
        title=f"Frozen Lake {length}x{length} Q-Learning Epsilon Greedy Time"
    plt.plot(epsilon,time_fl,linewidth=3.0,color="r")
    plt.xlabel('Epsilon')
    plt.grid()
    plt.title(title)
    
    plt.ylabel('Time (s)')
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()
        
        
    gammas=[0.95, 0.96, 0.97, 0.98, 0.99]
    time_fl=[]
    sz_arr_all=[]
    average_reward_all=[]
    global_rewards=[]
    staggered=[]
    for gamma in gammas:
        np.random.seed(0)
        this_map = generate_random_map(size=length)
        env = gym.make("FrozenLake-v1", desc=this_map)
        # env = gym.make("FrozenLake-v1")
        env = env.unwrapped
        desc = env.unwrapped.desc
        env.render()  # check environment
        num_episodes=10000
        start = time.time()
        Q_qlearning,local_rewards = Q_learning(env, lr=.1, num_episodes=10000, eps=.75, gamma=gamma, eps_decay=0.001)
        end = time.time()
        # sz_arr_all.append(sz_arr)
        # average_reward_all.append(average_reward)
        time_fl.append(end-start)
        global_rewards.append(local_rewards)
        
        size = int(num_episodes / 50)
        chunks = list(chunk_list(local_rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        sz_arr_all.append(size)
        staggered.append(chunks)
        average_reward_all.append(averages)
        
    print(sz_arr_all[0])
    for i, gamma in enumerate(gammas):    
        plt.plot(range(0, len(global_rewards[i]), sz_arr_all[i]), average_reward_all[i],label='gamma='+str(gamma))

    if greedy_:
        title=f"Frozen Lake {length}x{length} Q-Learning Greedy Varying Gamma Reward"
    else:
        title=f"Frozen Lake {length}x{length} Q-Learning Epsilon Greedy Varying Gamma Reward"
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    plt.legend(loc=2, prop={'size': 6})
    plt.ylabel('Reward')
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()

    if greedy_:
        title=f"Frozen Lake {length}x{length} Q-Learning Greedy Time Varying Gamma"
    else:
        title=f"Frozen Lake {length}x{length} Q-Learning Epsilon Greedy Time Varying Gamma"
    plt.plot(gammas,time_fl, linewidth=3.0,color="g")
    plt.xlabel('Gamma')
    
    plt.grid()
    plt.title(title)
    plt.ylabel('Time (s)')
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()


if __name__ == '__main__':
    run(20, False)
    run(20, True)
    run(4, False)
    run(4, True)
    # run(4)
    
