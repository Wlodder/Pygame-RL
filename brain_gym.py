import gym
import tensorflow
import numpy as np
import keras
import DQN
import matplotlib.pyplot as plt

# constant variables
explore_threshold = 0.75
explore_decay_rate = 0.00005
explore_min = 0.0
number_episodes = 4000

# setup environment and network
env = gym.make("Acrobot-v1")
model = DQN.q_network(env.observation_space.shape[0],env.action_space.n)
observation = env.reset()
rewards = []


for episode in range(number_episodes):
    # cumulative reward per episode to be plotted in graph
    episode_reward = 0
    print("starting episode " + str(episode  + 1))
    # time frame for episode to add terminal state
    for time in range(500):
        env.render()

        # chooses whether exploit or explore action will be taken
        if np.random.rand() < max(explore_threshold,explore_min):
            action = env.action_space.sample()
        else:
            action = model.predict(observation.reshape(1,env.observation_space.shape[0]))

        # gets information from environment and agent takes action
        new_observation, reward, done, info = env.step(action)

        # add the state and next state to replay memory
        model.add_to_memories(observation.reshape(1,env.observation_space.shape[0]), reward, done, new_observation.reshape(1,env.observation_space.shape[0]),action)

        observation = new_observation
        
        # check for terminal state
        if done:
            observation = env.reset()
            break

        episode_reward += reward
    
    # train the model 
    model.train(env.observation_space.shape[0],env.action_space.n,batch_ratio=0.4)
    
    # update weights of the target model in the DQN
    if episode % 100 == 0:
        model.update_target_model()

    # update the explore threshold 
    explore_threshold -= explore_decay_rate


    rewards.append(episode_reward)

env.close()
plt.plot(np.arange(number_episodes),np.array(rewards))
plt.xlabel('episode')
plt.ylabel('reward total')
plt.show()