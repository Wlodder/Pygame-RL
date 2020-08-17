import tensorflow as tf
import keras
import numpy as np
import random

replay_buffer_size = 10000

class q_network:

    gamma = 0.999
    explore_threshold = 0.5
    
    def __init__(self,input_size,output_size):
        self.model = self.create_model(input_size,output_size)
        self.target_model = self.create_model(input_size,output_size)
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.replay_memories = []


    def create_model(self, input_size,action_space_size):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24,input_shape=(input_size,),activation='relu'))
        model.add(keras.layers.Dense(24,activation='relu'))
        model.add(keras.layers.Dense(36,activation='relu'))
        model.add(keras.layers.Dense(36,activation='relu'))
        model.add(keras.layers.Dense(24,activation='relu'))
        model.add(keras.layers.Dense(action_space_size,activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='mse',metrics=['mae'])
        return model

    def add_to_memories(self,observation, reward, done, next_observation, action):
        if len(self.replay_memories) < replay_buffer_size:
            self.replay_memories.append((observation, reward, done, next_observation, action))
        else:
            self.replay_memories.pop(0)
            self.replay_memories.append((observation, reward, done, next_observation, action))

    def predict(self, observation):
        return np.argmax(self.target_model.predict(observation)[0])


    def train(self,number_of_observations, number_of_actions, batch_size=100, batch_ratio=0.1):
        
        if batch_size > len(self.replay_memories):
            size = int(batch_ratio * len(self.replay_memories))
        else:
            size = max(batch_size, int(batch_ratio * len(self.replay_memories)))

        if size == 0 or len(self.replay_memories) == 0:
            return

        mini_batch = random.sample(self.replay_memories,size)

        X = np.zeros(shape=(size,number_of_observations))
        y = np.zeros(shape=(size,number_of_actions))
        
        i = 0
        for memory in mini_batch:
            new_state = memory[3]
            current_state = memory[0]
            a = memory[4]
            reward = memory[1]
            done = memory[2]
            if not done:
                target = reward + self.gamma * np.max(self.target_model.predict(new_state))
            else:
                target = reward
            target_vec = self.model.predict(current_state)[0]
            target_vec[a] = target
            X[i] = current_state
            y[i] = target_vec
            i += 1
        self.model.fit(X,y,verbose=0)
        self.replay_memories.clear()
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
