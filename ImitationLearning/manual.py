"""
You don't have to edit this script.
You may, however change the value of MAX_STEPS.
"""
MAX_STEPS = 10000

import gym 
#from pyglet.window import key
import numpy as np 
import pickle
import os 
from datetime import datetime
import gzip
import json

# From CarRacing-v0's Documentation
def key_press(k, mod):
    # global restart
    # if k == 0xFF0D:
    #     restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0

def store_data(data, directory = 'expert_data'):
    if not os.path.exists(directory):
        os.mkdir(directory)
    data_file = os.path.join(directory, 'data.pkl.gzip')
    with gzip.open(data_file, 'wb') as file:
        pickle.dump(data, file)

    print(f"Data stored in {data_file}")
    
def store_results(episode_rewards, directory = 'results'):
    if not os.path.exists(directory):
        os.mkdir(directory)

    results = {}
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    results_file = os.path.join(directory, "manual-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(results_file, 'w') as file:
        json.dump(results, file)
        
    print(f"Results stored in {results_file}.")

if __name__ == "__main__":

    env = gym.make('CarRacing-v0', verbose = 1).unwrapped # Turn to 0 to stop printing track info

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0.0, 0.0, 0.0]).astype('float32')
    
    episode_rewards = []
    steps = 0
    states, actions, next_states, rewards, is_dones = [] , [], [], [], []

    data_samples = {
        "state" : [],
        "action" : [],
        "reward" : [],
        "next_state" : [],
        "terminal" : []
    }

    while steps < MAX_STEPS:
        episode_reward = 0
        state = env.reset()
        done = False

        while not done:
            next_state, r, done, info = env.step(a)
            episode_reward += r

            data_samples['state'].append(state)     
            data_samples['action'].append(np.array(a))
            data_samples['next_state'].append(next_state)
            data_samples['reward'].append(r)
            data_samples['terminal'].append(done)
            
            state = next_state
            steps += 1

            if steps % 5000 == 0:
                print('... saving data')
                store_data(data_samples, "./data")
                store_results(episode_rewards, "./results")

            env.render()
        
        episode_rewards.append(episode_reward)

    env.close()
