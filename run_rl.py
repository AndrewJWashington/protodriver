import os
import random
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import pydirectinput
import keyboard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from collections import deque  # todo - switch back? otherwise will use extra memory

from protodriver import utils


#config
COUNT_DOWN = True
MAX_FRAMES = 501 # none for infinite runtime, roughly 10 fps for training and 1.5 fps for running
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOAD_MODEL = True
MODEL_FILENAME = 'rl_model'
TARGET_MODEL_FILENAME = 'rl_target_model'


#todo - move class elsewhere
# class adapted from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
class DQN:
    def __init__(self, model_filename=None, target_model_filename=None):
        self.input_shape = (75, 100, 3)
        self.batch_input_shape = (-1, 75, 100, 3)      
        self.num_actions = 8  # forward, forward left, left, ...
        
        self.memory = list()
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        if model_filename is None:
            self.model = self.create_model()
        else:
            self.model = keras.models.load_model(f'models/{model_filename}')
            self.model.summary()
        
        if model_filename is None:
            self.target_model = self.create_model()
        else:
            self.target_model = keras.models.load_model(f'models/{target_model_filename}')
            self.target_model.summary()

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(filters=10, input_shape=self.input_shape,
                         kernel_size=(15, 15), strides=(5, 5), padding="valid",
                         activation = "relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
        model.add(layers.Conv2D(filters=10, kernel_size=(5,5), strides=(1,1), padding="same", activation = "relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
        model.add(layers.Flatten())
        model.add(layers.Dense(units = 10, activation = "relu"))
        model.add(layers.Dense(self.num_actions))  # output layer
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.summary()
        return model

    def act(self, state):
        #print('Acting...')
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            #print('random action')
            return np.floor(np.random.rand() * self.num_actions)
        #print('action from model')
        #print('state', state.reshape(1, 16, 1, 1))
        prediction = self.model.predict(state.reshape(self.batch_input_shape))[0]
        #print('prediction', prediction)
        #print('action', np.argmax(prediction))
        return np.argmax(prediction)  # need to change this to allow for no input

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        #print('Replaying...')
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state.reshape(self.batch_input_shape))

            if done:
                target[0][int(action)] = reward
            else:
                prediction = self.target_model.predict(state.reshape(self.batch_input_shape))[0]
                Q_future = max(prediction)
                #print('prediction', prediction)
                #print('Q_future', Q_future)
                #print('action', action)

                # will need some work here to store and access the 4x1 output vector
                # possibly store as dictionary instead of deque
                target[0][int(action)] = reward + Q_future * self.gamma
            self.model.fit(state.reshape(self.batch_input_shape), target, epochs=1, verbose=0)

    def target_train(self):
        #print('Training target model...')
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_models(self, model_filename, target_model_filename):
        self.model.save(f'models/{model_filename}')
        self.target_model.save(f'models/{target_model_filename}')
        print(f"Saved models to models/{model_filename} and models/{target_model_filename}")


if __name__ == "__main__":
    print("running")

    # countdown
    if COUNT_DOWN:
        for count in range(3, 0, -1):
            print(count)
            time.sleep(0.5)

    # init
    frames_processed = 0
    user_exit = False
    last_time = time.time()
    if MAX_FRAMES is None:
        MAX_FRAMES = int("inf")
    reward_obj = utils.Reward()
        
    if LOAD_MODEL:
        dqn_agent = DQN(MODEL_FILENAME, TARGET_MODEL_FILENAME)
    else:
        dqn_agent = DQN()
        
    # grab screen to initiate necessary variables
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    last_processed_screen = utils.process_image(screen)
    last_flow = np.zeros_like(last_processed_screen)

    # game loop
    while frames_processed < MAX_FRAMES and not user_exit:
        done = False
        frame_number = frames_processed  # todo - make name of frame independent of how many processed
        
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))

        # process image and display resulting image
        processed_screen = utils.process_image(screen)
        cv2.imshow('window', processed_screen)

        user_input = utils.get_user_input()
        if user_input[4]:  # space pressed
            user_exit = True
            
        # get model prediction 
        model_input = np.array(processed_screen).reshape((1, 75, 100, 3))
        prediction = dqn_agent.act(model_input)
        #prediction_str = " ".join([f"{p:2.2}" for p in prediction])

        # send input
        utils.send_input_single_key(prediction)
        
        # some stuff to get opencv not to crash
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        # learn
        flow_scalar, last_flow = utils.calculate_optical_flow(last_processed_screen,
                                                              processed_screen,
                                                              last_flow)
        reward = reward_obj.get_reward(flow_scalar, prediction)
        dqn_agent.remember(last_processed_screen, prediction, reward, processed_screen, done)
        dqn_agent.replay()
        dqn_agent.target_train()
        last_processed_screen = processed_screen

        print('Predicted:', prediction)
        print(f'Reward: {reward:3.3} (flow: {flow_scalar:3.3})')

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        frames_processed = frames_processed + 1
        print(f"Framerate: {fps:4.4} fps, ({frames_processed} / {MAX_FRAMES}) frames processed")

        if frames_processed % 50 == 0:
            dqn_agent.save_models(MODEL_FILENAME, TARGET_MODEL_FILENAME)
        
    # feet off the pedals!
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('s')
    
    print("completed successfully")
