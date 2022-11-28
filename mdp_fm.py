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
import mdptoolbox, mdptoolbox.example
import hiive




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


#https://miat.inrae.fr/MDPtoolbox/QuickStart.pdf
#https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
   
def forest_PI(states):
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    P, R = mdptoolbox.example.forest(S=states)

    score=[]
    timing=[]
    



    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fm_policy = []
    runtime_fm_policy = []
    fm_iters=[]
    rewards=[]
    policies=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,21):
        
        
        # policy_iteration(env)
        gamma=i/20
        if gamma==1:
            gamma=.99
        gamma_fm_policy.append(gamma)
        start = time.time()
        pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        pi.run()
        end = time.time()
        runtime_fm_policy.append(end-start)
        rewards.append(np.mean(pi.V)) 
        policies.append(pi.policy)
        fm_iters.append(pi.iter)

    plt.plot(gamma_fm_policy, runtime_fm_policy,linewidth=3.0,color="g")
    plt.xlabel('Gamma')
    plt.title('Forest Management ' + str(states) + 'Runtime (Policy Iteration)')
    plt.ylabel('Time (s)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Runtime (Policy Iteration).png')  
    plt.clf()

    
    plt.plot(gamma_fm_policy,rewards,linewidth=3.0,color="b")
    plt.xlabel('Gamma')
    plt.ylabel('Rewards')
    plt.title('Forest Management ' + str(states) + ' States Rewards (Policy Iteration)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Rewards (Policy Iteration)')  
    plt.clf()

    plt.plot(gamma_fm_policy,fm_iters,linewidth=3.0,color="r")
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.title('Forest Management ' + str(states) + ' States Iterations to Convergence (Policy Iteration)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Iterations to Convergence (Policy Iteration).png')  
    plt.clf()

def forest_VI(states):
    # grid_ = generate_random_map(5)
    # env = gym.make('FrozenLake-v0')
    np.random.seed(4)
    P, R = mdptoolbox.example.forest(S=states)

    score=[]
    timing=[]
    



    # print(this_map)
    # print(desc)
    # print(env)
    gamma_fm_policy = []
    runtime_fm_policy = []
    fm_iters=[]
    rewards=[]
    policies=[]
    # policy_iteration(env, eon, ean, gamma=0.99)
    # print(policy_iteration(env,eon,ean))
    worst_policy=[None,math.inf,0]
    best_policy=[None,-math.inf,0]
    for i in range(1,21):
        
        
        # policy_iteration(env)
        gamma=i/20
        if gamma==1:
            gamma=.99
        gamma_fm_policy.append(gamma)
        start = time.time()
        pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        pi.run()
        end = time.time()
        runtime_fm_policy.append(end-start)
        rewards.append(np.mean(pi.V)) 
        policies.append(pi.policy)
        fm_iters.append(pi.iter)

    plt.plot(gamma_fm_policy, runtime_fm_policy,linewidth=3.0,color="g")
    plt.xlabel('Gamma')
    plt.title('Forest Management ' + str(states) + 'States Runtime (Value Iteration)')
    plt.ylabel('Time (s)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Runtime (Value Iteration).png')  
    plt.clf()

    
    plt.plot(gamma_fm_policy,rewards,linewidth=3.0,color="b")
    plt.xlabel('Gamma')
    plt.ylabel('Rewards')
    plt.title('Forest Management ' + str(states) + ' States Rewards (Value Iteration)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Rewards (Value Iteration)')  
    plt.clf()

    plt.plot(gamma_fm_policy,fm_iters,linewidth=3.0,color="r")
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.title('Forest Management ' + str(states) + ' States Iterations to Convergence (Value Iteration)')
    plt.grid()
    plt.savefig('./plots/Forest Management ' + str(states) + 'Iterations to Convergence (Value Iteration).png')  
    plt.clf()



if __name__ == "__main__":
    forest_PI(16)
    forest_PI(400)
    forest_VI(16)
    forest_VI(400)
