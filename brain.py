import tensorflow as tf
import keras
import numpy as np
import pygame
import crop
import farmer_class
import pygame
import heapq
from itertools import count


class our_model:

    gamma = 0.75
    explore_threshold = 0.5
    reward_avg_list = []
    screen_x = 1000
    screen_y = 1000
    farmer_start_x = 100
    farmer_start_y = 100

    def explore_function(self, i):
        return float(500 / i)

    def __init__(self,x,y):
        self.model = self.create_model(x,y)
        self.file = open("reward_log.txt","w+")
        self.target_model = self.create_model(x,y)
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.replay_memories = []
        self.training_number = 1

    def get_state(self, farmer, screen):
        size = farmer.visionRect.height * farmer.visionRect.width
        state = np.zeros(shape=(1,size))
        z = 0
        mini_state = np.zeros(shape=(1,9))
        for x in range(farmer.visionRect.x - int(farmer.visionRect.width/2), farmer.visionRect.x + int(farmer.visionRect.width/2)):
                for y in range(farmer.visionRect.y - int(farmer.visionRect.height/2), farmer.visionRect.y + int(farmer.visionRect.height/2)):
                    if x < self.screen_x and x > 0 and y < self.screen_y and y > 0 and screen.get_at((x,y)) == ((0,0,0)):
                        state[0,z] = 1
                    elif not(x < self.screen_x and x > 0 and y < self.screen_y and y > 0):
                        state[0,z] = -1
                    z += 1
        done = farmer.life > 100
        for i in range(9):
            mini_state[0,i] = np.mean(state[0,int(size / 9 * (i)):int(size / 9 * (i + 1))]) * 1000

        if not(farmer.x < self.screen_x and farmer.x > 0 and farmer.y < self.screen_y and farmer.y > 0):
            done = True


        return mini_state, done

    def reward_function(self, farmer, crops, time):
        reward = 0
        terminate_state = False
        for grain in crops:
            if farmer.colliderect(grain):
                crops.remove(grain)
                crop.plant_crop(crops)
                reward += 100000
        if farmer.x < 0 or farmer.x > self.screen_x or farmer.y < 0 or farmer.y > self.screen_y:
            terminate_state = True
            reward -= 100000

    
        if time % 500 == 0:
            crop.plant_crop(crops)
        return reward, terminate_state

    def create_model(self, x, y):
        model = keras.Sequential()
        model.add(keras.layers.Dense(50,input_shape=(x*y,)))
        model.add(keras.layers.Dense(4,activation='relu'))
        model.compile(optimizer='adam',loss='mse',metrics=['mae'])
        return model

    def run(self):
        pygame.init()
        size = (self.screen_x,self.screen_y)
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("My First Game")
        farmer = farmer_class.farmer(self.farmer_start_x,self.farmer_start_y,100,100)
        farmer.defineVision(100,100)
        
        for i in range(1,10000):
            crops = []
            farmer.x = self.farmer_start_x
            farmer.y = self.farmer_start_y
            farmer.visionRect.x = self.farmer_start_x
            farmer.visionRect.y = self.farmer_start_y
            crop.plant_crop(crops)
            current_state,_ = self.get_state(farmer,screen)
            r_sum = 0
            done = False
            reward = 0
            j = 0
            farmer.life = 0
            
            while not done:
                pending_done = False
                pygame.event.pump()
                j += 1
                threshold = self.explore_function(i)
                
                if np.random.random() < threshold:
                    a = np.random.randint(0,4)
                    farmer.process(a)
                    reward, pending_done = self.reward_function(farmer, crops, j)
                else:
                    a = np.argmax(self.model.predict(current_state)[0])
                    farmer.process(a)
                    reward,pending_done = self.reward_function(farmer, crops, j)
                ### where the game is run
                
                screen.fill((255,255,255))
                pygame.draw.rect(screen,farmer.color,farmer)
                for grain in crops:
                    pygame.draw.rect(screen,grain.color,grain)
                pygame.display.flip()
            
                ###
                self.explore_threshold += 0.0001
                new_state, pending_done = self.get_state(farmer,screen)
                if pending_done:
                    reward = -100000
                if len(self.replay_memories) < 1000000:
                    self.replay_memories.append((reward,a,current_state,new_state))
                else:
                    self.replay_memories.pop(0)
                    self.replay_memories.append((reward,a,current_state,new_state))
                
                current_state = new_state
                r_sum += reward
                
                self.target_update_counter += 1
                if self.target_update_counter % 1500 == 0:
                    print("training session " + str(self.training_number))
                    self.train()
                    self.training_number += 1
                done = pending_done
            self.reward_avg_list.append(r_sum)
        print(self.reward_avg_list)
        pygame.quit()
        self.file.close()

    def train(self):
        
        num_samples = int(len(self.replay_memories) * 0.001)
        self.replay_memories.sort(key=memory_sort_helper)
        
        training_array = []
        for elem in self.replay_memories:
            if elem[0] != 0:
                training_array.append(elem)
        memories_interesting = np.array(training_array).reshape(len(training_array),4)
        
        j = 0
        for memory in memories_interesting:
            self.file.write("memory "  + str(j) + " : ")
            self.file.write("reward " + str(memory[0]) + "\n")
            j += 1
        
        memories_rewarded_highest = np.array(self.replay_memories[0:num_samples]).reshape(num_samples,4)
        memories_rewarded_smallest = np.array(self.replay_memories[(len(self.replay_memories)-num_samples):len(self.replay_memories)]).reshape(num_samples,4)
        memories_random_index = np.random.choice(len(self.replay_memories), num_samples, replace=False)
        memories_random = np.array(np.array(self.replay_memories)[memories_random_index]).reshape(num_samples,4)
        memories_total = np.append(np.append(memories_random,memories_rewarded_highest,axis=1),memories_rewarded_smallest,axis=1)
        memories_total = memories_interesting
        X = np.zeros(shape=(len(memories_total),9))
        y = np.zeros(shape=(len(memories_total),4))
        print("training sessions " + str(self.training_number) +  " rewards : ")

        for i in range(len(memories_total)):
            new_state = memories_total[i][3]
            current_state = memories_total[i][2]
            a = memories_total[i][1]
            reward = memories_total[i][0]
            target = reward + self.gamma * np.max(self.target_model.predict(new_state))
            target_vec = self.model.predict(current_state)[0]
            target_vec[a] = target
            X[i] = current_state
            y[i] = target_vec
        self.model.fit(X,y,epochs=10,verbose=0,batch_size=len(memories_total))
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memories.clear()


def memory_sort_helper(memory):
    return memory[0]