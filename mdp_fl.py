# import mlrose_hiive as mlrose
from cmath import inf
from telnetlib import GA
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
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


color= {
        b'S': 'blue',
        b'F': 'gray',
        b'H': 'black',
        b'G': 'red',
    }


state= {
        b'S': 'S',
        b'F': 'F',
        b'H': 'H',
        b'G': 'G',
    }

direction= {
        3: 'U',
        2: 'R',
        1: 'D',
        0: 'L'
    }




#https://people.math.ethz.ch/~jteichma/lecture_7.html
def run_episode(env, policy, gamma, render = True):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs,info = env.reset()
    # print(obs)
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        # print(int(policy[obs]))
        obs, reward, done ,___, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


#https://people.math.ethz.ch/~jteichma/lecture_7.html
def evaluate_policy(env, policy, gamma,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [run_episode(env, policy, gamma = gamma, render = False) for _ in range(n)]
    return np.mean(scores)

# https://www.kaggle.com/code/jinbonnie/reinforcementlearning-frozenlake-cartpole
def compute_value_function(policy, env, gamma, eon):
    eon = env.observation_space.n
    ean = env.action_space.n
    value_table = np.zeros(eon)
    # set the threshold
    threshold = 1e-10
    #set threshold really near to zero

    #stop until convergence
    while True:
        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)

        # for each state in the environment, 
        #select the action according to the policy and compute the value table
        for state in range(eon):
            action = policy[state]
  
            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                        for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        #see if the formula convergence
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    #value-table
    return value_table


def extract_policy(value_table, env, gamma = 0.99):
    
    # Initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):

        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)

        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # Select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

#https://people.math.ethz.ch/~jteichma/lecture_7.html
def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v,i+1



def policy_iteration(env, eon, ean, gamma=0.99):
    
    # Initialize policy with zeros
    old_policy = random_policy = np.ones(eon)
    no_of_iterations = 200000

    for i in range(no_of_iterations):

        # compute the value function
        new_value_function = compute_value_function(old_policy, env, gamma, eon)
        # compute_value_function(policy, env, gamma, eon)

        # Extract new policy from the computed value function
        new_policy = extract_policy(new_value_function,  env, gamma)
        # extract_policy(value_table, env, eon, ean, gamma = 0.99)

        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing old_policy and new policy if it same we will break the iteration
        # else we update old_policy with new_policy

        if (np.all(old_policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        old_policy = new_policy

    return new_policy, i+1

# print(policy_iteration(env))

# def policy_iteration():
#     pass

# def value_iteration():
#     pass

# def q_learning():
#     pass
    
def frozen_lake_PI_20():
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    this_map = generate_random_map(size=20)
    env = gym.make("FrozenLake-v1", desc=this_map)
    # env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    desc = env.unwrapped.desc
    env.render()  # check environment




    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, 20), ylim=(0, 20))

    font_size='small'
    title = "Frozen Lake 20x20 Environment"
    plt.title(title)
    for i in range(20):
        for j in range(20):
            y = 20 - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, state[desc[i,j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    plt.tight_layout()
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()
    


    eon = env.observation_space.n
    ean = env.action_space.n

    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fl_policy = []
    runtime_fl_policy = []
    fl_iters=[]
    rewards=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,21):
        
        
        # policy_iteration(env)
        gamma=i/20
        gamma_fl_policy.append(gamma)
        start = time.time()
        policy, iteration = policy_iteration(env, eon, ean, gamma)
        end = time.time()
        runtime_fl_policy.append(end-start)
        fl_iters.append(iteration)
        # policy = extract_policy(optimal_v, gamma)
        # print("This is policy: ",policy)
        policy_score = evaluate_policy(env, policy, gamma, n=1000)
        print(f"This is policy score: {policy_score}")
        print('Average scores = ', np.mean(policy_score))
        if policy_score>best_policy[1]:
            best_policy=[policy,policy_score,gamma]
            
        if policy_score<worst_policy[1]:
            worst_policy=[policy,policy_score,gamma]
        
        rewards.append(policy_score)
        
        print(f"Took this many iterations: {iteration} for gamma = {gamma}")
        

    policy = best_policy[0].reshape(20,20)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 20x20 for gamma = "+str(best_policy[2]) +" (Policy Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()
    plt.savefig("./plots/"+title+"20x20" +str('.png'))
    plt.close()
    
    
    policy = worst_policy[0].reshape(20,20)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 20x20 for gamma = "+str(worst_policy[2]) + " (Policy Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()

    plt.savefig("./plots/"+title+"20x20" +str('.png'))
    plt.close()
    
    #     gamma_fl_policy = []
    # runtime_fl_policy = []
    # fl_iters=[]
    # Plotting the Graphs
    plt.plot(gamma_fl_policy, fl_iters,linewidth=3.0, color="r")
    plt.grid()
    plt.title("Frozen Lake 20x20 Iterations to Convergence (Policy Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Iterations")
    plt.savefig('./plots/gamma_iters_fl_20x20_Policy_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, runtime_fl_policy,linewidth=3.0, color="g")
    plt.grid()
    plt.title("Frozen Lake 20x20 Runtime (Policy Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Time (s)")
    plt.savefig('./plots/gamma_time_fl_20x20_Policy_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, rewards,linewidth=3.0, color="b")
    plt.grid()
    plt.title("Frozen Lake 20x20 Rewards (Policy Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Rewards")
    plt.savefig('./plots/gamma_rewards_fl_20x20_Policy_Iteration.png')  
    plt.clf()    

def frozen_lake_PI_4():
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    this_map = generate_random_map(size=4)
    env = gym.make("FrozenLake-v1", desc=this_map)
    # env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    desc = env.unwrapped.desc
    env.render()  # check environment




    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, 4), ylim=(0, 4))

    font_size='small'
    title = "Frozen Lake 4x4 Environment"
    plt.title(title)
    for i in range(4):
        for j in range(4):
            y = 4 - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, state[desc[i,j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, 4))
    plt.ylim((0, 4))
    plt.tight_layout()
    plt.savefig("./plots/"+title+str('.png'))
    plt.close()
    


    eon = env.observation_space.n
    ean = env.action_space.n

    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fl_policy = []
    runtime_fl_policy = []
    fl_iters=[]
    rewards=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,20):
        
        
        # policy_iteration(env)
        gamma=i/20
        gamma_fl_policy.append(gamma)
        start = time.time()
        policy, iteration = policy_iteration(env, eon, ean, gamma)
        end = time.time()
        runtime_fl_policy.append(end-start)
        fl_iters.append(iteration)
        # policy = extract_policy(optimal_v, gamma)
        # print("This is policy: ",policy)
        policy_score = evaluate_policy(env, policy, gamma, n=1000)
        print(f"This is policy score: {policy_score}")
        print('Average scores = ', np.mean(policy_score))
        if policy_score>best_policy[1]:
            best_policy=[policy,policy_score,gamma]
            
        if policy_score<worst_policy[1]:
            worst_policy=[policy,policy_score,gamma]
        
        rewards.append(policy_score)
        
        print(f"Took this many iterations: {iteration} for gamma = {gamma}")
        

    policy = best_policy[0].reshape(4,4)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 4x4 for gamma = "+str(best_policy[2]) + " (Policy Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()
    plt.savefig("./plots/"+title+"4x4" +str('.png'))
    plt.close()
    
    
    policy = worst_policy[0].reshape(4,4)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 4x4 for gamma = "+str(worst_policy[2]) + " (Policy Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()

    plt.savefig("./plots/"+title+"4x4" +str('.png'))
    plt.close()
    
    #     gamma_fl_policy = []
    # runtime_fl_policy = []
    # fl_iters=[]
    # Plotting the Graphs
    plt.plot(gamma_fl_policy, fl_iters,linewidth=3.0, color="r")
    plt.grid()
    plt.title("Frozen Lake 4x4 Iterations to Convergence (Policy Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Iterations")
    plt.savefig('./plots/gamma_iters_fl_4x4_Policy_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, runtime_fl_policy,linewidth=3.0, color="g")
    plt.grid()
    plt.title("Frozen Lake 4x4 Runtime (Policy Iteration")
    plt.xlabel("Gamma")
    plt.ylabel("Time (s)")
    plt.savefig('./plots/gamma_time_fl_4x4_Policy_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, rewards,linewidth=3.0, color="b")
    plt.grid()
    plt.title("Frozen Lake 4x4 Rewards (Policy Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Rewards")
    plt.savefig('./plots/gamma_rewards_fl_4x4_Policy_Iteration.png')  
    plt.clf()  
    
def frozen_lake_VI_20():
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    this_map = generate_random_map(size=20)
    env = gym.make("FrozenLake-v1", desc=this_map)
    # env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    desc = env.unwrapped.desc
    env.render()  # check environment

    eon = env.observation_space.n
    ean = env.action_space.n

    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fl_policy = []
    runtime_fl_policy = []
    fl_iters=[]
    rewards=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,20):
        
        
        # policy_iteration(env)
        gamma=i/20
        gamma_fl_policy.append(gamma)
        start = time.time()
        optimal_v , iteration = value_iteration(env,gamma)
        # optimal_v = value_iteration(env, gamma);
        
        end = time.time()
        runtime_fl_policy.append(end-start)
        policy = extract_policy(optimal_v, env,gamma)
        fl_iters.append(iteration)
        # policy = extract_policy(optimal_v, gamma)
        # print("This is policy: ",policy)
        policy_score = evaluate_policy(env, policy, gamma, n=1000)
        print(f"This is policy score: {policy_score}")
        print('Average scores = ', np.mean(policy_score))
        if policy_score>best_policy[1]:
            best_policy=[policy,policy_score,gamma]
            
        if policy_score<worst_policy[1]:
            worst_policy=[policy,policy_score,gamma]
        
        rewards.append(policy_score)
        
        print(f"Took this many iterations: {iteration} for gamma = {gamma}")
        
    # return
    policy = best_policy[0].reshape(20,20)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 20x20 for gamma = "+str(best_policy[2])+" (Value Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            
    
    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()
    plt.savefig("./plots/"+title+"20x20" +str('.png'))
    plt.close()
    
    
    policy = worst_policy[0].reshape(20,20)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 20x20 for gamma = "+str(worst_policy[2])+ " (Value Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()

    plt.savefig("./plots/"+title+"20x20" +str('.png'))
    plt.close()
    
    #     gamma_fl_policy = []
    # runtime_fl_policy = []
    # fl_iters=[]
    # Plotting the Graphs
    plt.plot(gamma_fl_policy, fl_iters,linewidth=3.0, color="r")
    plt.grid()
    plt.title("Frozen Lake 20x20 Iterations to Convergence (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Iterations")
    plt.savefig('./plots/gamma_iters_fl_20x20_Value_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, runtime_fl_policy,linewidth=3.0, color="g")
    plt.grid()
    plt.title("Frozen Lake 20x20 Runtime (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Time (s)")
    plt.savefig('./plots/gamma_time_fl_20x20_Value_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, rewards,linewidth=3.0, color="b")
    plt.grid()
    plt.title("Frozen Lake 20x20 Rewards (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Rewards")
    plt.savefig('./plots/gamma_rewards_fl_20x20_Value_Iteration.png')  
    plt.clf()   
     
def frozen_lake_VI_4():
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    this_map = generate_random_map(size=4)
    env = gym.make("FrozenLake-v1", desc=this_map)
    # env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    desc = env.unwrapped.desc
    env.render()  # check environment

    eon = env.observation_space.n
    ean = env.action_space.n

    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fl_policy = []
    runtime_fl_policy = []
    fl_iters=[]
    rewards=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,21):
        
        
        # policy_iteration(env)
        gamma=i/20
        gamma_fl_policy.append(gamma)
        start = time.time()
        optimal_v , iteration = value_iteration(env,gamma)
        # optimal_v = value_iteration(env, gamma);
        
        end = time.time()
        runtime_fl_policy.append(end-start)
        policy = extract_policy(optimal_v, env,gamma)
        fl_iters.append(iteration)
        # policy = extract_policy(optimal_v, gamma)
        # print("This is policy: ",policy)
        policy_score = evaluate_policy(env, policy, gamma, n=1000)
        print(f"This is policy score: {policy_score}")
        print('Average scores = ', np.mean(policy_score))
        if policy_score>best_policy[1]:
            best_policy=[policy,policy_score,gamma]
            
        if policy_score<worst_policy[1]:
            worst_policy=[policy,policy_score,gamma]
        
        rewards.append(policy_score)
        
        print(f"Took this many iterations: {iteration} for gamma = {gamma}")
        
    # return
    policy = best_policy[0].reshape(4,4)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 4x4 for gamma = "+str(best_policy[2])+"(Value Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            
    
    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()
    plt.savefig("./plots/"+title+"4x4" +str('.png'))
    plt.close()
    
    
    policy = worst_policy[0].reshape(4,4)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size='small'
    title = "Frozen Lake 4x4 for gamma = "+str(worst_policy[2])+" (Value Iteration)"
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1,edgecolor='white')
            p.set_facecolor(color[desc[i,j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction[policy[i, j]], weight='bold', size=font_size,
                        horizontalalignment='center', verticalalignment='center', color='w')
            

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    # plt.tight_layout()

    plt.savefig("./plots/"+title+"4x4" +str('.png'))
    plt.close()
    
    #     gamma_fl_policy = []
    # runtime_fl_policy = []
    # fl_iters=[]
    # Plotting the Graphs
    plt.plot(gamma_fl_policy, fl_iters,linewidth=3.0, color="r")
    plt.grid()
    plt.title("Frozen Lake 4x4 Iterations to Convergence (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Iterations")
    plt.savefig('./plots/gamma_iters_fl_4x4_Value_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, runtime_fl_policy,linewidth=3.0, color="g")
    plt.grid()
    plt.title("Frozen Lake 4x4 Runtime (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Time (s)")
    plt.savefig('./plots/gamma_time_fl_4x4_Value_Iteration.png')  
    plt.clf()
    
    plt.plot(gamma_fl_policy, rewards,linewidth=3.0, color="b")
    plt.grid()
    plt.title("Frozen Lake 4x4 Rewards (Value Iteration)")
    plt.xlabel("Gamma")
    plt.ylabel("Rewards")
    plt.savefig('./plots/gamma_rewards_fl_4x4_Value_Iteration.png')  
    plt.clf()    
    
# def epsilon_greedy(q_table, s, num_episodes):
#     #for each time step within an episode we set our exploration rate threshold
#     rand_num = rd.randint(0, 20000)
#     if rand_num > num_episodes:
#         #the agaent will explore the environment and simple action randomly
#         action = rd.randint(0, 3)
#     else :
#         #the agent will exploit the environment and choose the action
#         #has the highest key value in the Q-table for the current state
#         action = np.argmax(q_table[s, :])
#     return action

def greedy(q_table, s):

    action = np.argmax(q_table[s, :])
    return action

def epsilon_greedy(q_table, s, epsilon=0.05):
    #for each time step within an episode we set our exploration rate threshold
    rand_num = rd.randint(0, 20000)
    if np.random.uniform(0, 1) < (epsilon):
        #the agaent will explore the environment and simple action randomly
        action = rd.randint(0, 3)
    else :
        #the agent will exploit the environment and choose the action
        #has the highest key value in the Q-table for the current state
        action = np.argmax(q_table[s, :])
    return action

#https://www.kaggle.com/code/jinbonnie/frozenlake-reinforcement-q-learning/notebook    
def train(env, learningRate,discountFactor):
    Q = np.zeros([env.observation_space. n,env.action_space.n])
    for i_episodes in range(20000): ##for each time step within an episode
    #for each episode
    #we're going to first rest the state of the environment back to the starting state
      s,_ = env.reset()
      i = 0
    # Q-Table
      while i < 1000000:
        i += 1
        #use epsilon greedy strategy
        a = epsilon_greedy(Q, s, i_episodes)
        #new state, reward for that, whether the action in our episode, and diagnostic information
        observation, reward, done, info,_ = env.step(a)
        #update Q-table
         
        Q[s, a] = (1-learningRate) * Q[s, a] + learningRate * ( reward + discountFactor * np.max(Q[observation,:]))
        s = observation
        #'done' check whether or not our episode is finished
        if done:
            break 
    return Q
    
#contains everything that happens for a single timestamp within each spisode
def test(env,Q):
    for i_episodes in range(100):
        
        #reset environment
        rewardList=[]
        s,_ = env.reset()
        i = 0
        total_reward = 0
        while i < 500:
            i += 1

            #choose an action
            a = np.argmax(Q[s, :])

            #act new action and update
            observation, reward, done, info,_ = env.step(a)

            env.render()
            #update the rewards from current episode by adding the reward we received
            total_reward += reward
            s = observation
            if done:
                break
        rewardList.append(total_reward)
        
    return rewardList
        
def frozen_lake_Q_20():
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    # np.random.seed(4)
    this_map = generate_random_map(size=20)
    env = gym.make("FrozenLake-v1", desc=this_map)
    # env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    desc = env.unwrapped.desc
    env.render()  # check environment
    
    

    learningRate = 0.05
    discountFactor = 0.99
    rewardList = []
    
    epsilon = [0.05, 0.1, 0.125, 0.15, 0.25, 0.375, 0.5, 0.75, 0.90]

    Q=train(env, learningRate,discountFactor)
    rewardList=test(env,Q)
    
    print("This is reward  list len: ",len(rewardList))
    
    print("Success rate: " + str(sum(rewardList)/len(rewardList)))
if __name__ == "__main__":
    frozen_lake_PI_20()
    frozen_lake_PI_4()
    frozen_lake_VI_20()
    frozen_lake_VI_4()
    
