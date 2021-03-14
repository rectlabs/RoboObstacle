import time,os,sys, pygame
import numpy as np
import tensorflow as tf
from pygame.locals import *
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import load_model
from collections import deque
pygame.init()



class DQN:
    def __init__(self, size = 5000, inputnumbers = 10, outputnumbers = 10):
        self.inputnumbers = inputnumbers
        self.outputnumbers = outputnumbers
        self.learning_rate = 0.001
        self.momentum = 0.95
        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 2000000
        self.replay_memory_size = size
        self.replay_memory = deque([], maxlen=size)
        n_steps = 4000000  # total number of training steps
        self.training_start = 10000  # start training after 10,000 game iterations
        self.training_interval = 4  # run a training step every 4 game iterations
        self.save_steps = 1000  # save the model every 1,000 training steps
        self.copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
        self.discount_rate = 0.99
        # Skip the start of every game (it's just waiting time).
        self.skip_start = 90
        self.batch_size = 100
        self.iteration = 0  # game iterations
        self.done = True  # env needs to be reset
        self.recordReward = 0

        self.model = self.DQNmodel()

        
        return


    
    def stats(self):
        memory = list(self.replay_memory)
        sizeOfMemory = len(memory)
        self.recordReward += memory[-1][2]

        # print('Total memory: ', str(sizeOfMemory), ' , total rewards: ', str(self.recordReward))
        return

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.inputnumbers,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.outputnumbers, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def sample_memories(self, batch_size):
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        # state, action, reward, next_state, continue
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3].reshape(-1,1), cols[4].reshape(-1, 1))

    def epsilon_greedy(self, q_values, step):
        self.epsilon = max(self.eps_min, self.eps_max -
                           (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < self.epsilon:
            return np.random.randint(10)  # random action
        else:
            return np.argmax(q_values)  # optimal action


    



class RoboObstacle:
    def __init__(self, fps=50, storageSize = 5000, possibilities = 4, windowsize = (300,300)):
        # set up the window
        self.windowsize = windowsize
        self.DISPLAYSURF = pygame.display.set_mode(self.windowsize, 0, 32)
        pygame.display.set_caption('RoboObstacle')
        # set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.dict_shapes = {}
        self.shapes_state = []

        self.Agent = DQN(size = storageSize)
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()

        self.possibilities = possibilities

    def trainDQN(self):
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            self.Agent.sample_memories(self.Agent.batch_size))
        

        next_q_values = self.Agent.model.predict(X_state_val)
        max_next_q_values = np.max(
            next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * self.Agent.discount_rate * max_next_q_values

        # Train the online DQN
        # print('fitting: ', len(X_state_val), ' datas')
        self.Agent.model.fit(X_state_val, tf.keras.utils.to_categorical(
            X_next_state_val, num_classes=self.Agent.outputnumbers), verbose=0)


        return True

    def show_n_shapes(self, n, y_location, generate_shapes = True):
        if generate_shapes == True:
            self.dict_shapes = {}
            self.shapes_state = []
            # n varies from 0 to 7
            x_poses = [i for i in range(self.possibilities)]
            for m in range(n):
                x_position = np.random.choice(x_poses, replace = False)
                self.shapes_state.append(x_position)
                x_location = (x_position * 300)/ self.possibilities
                choice = np.random.choice(['rect', 'circle'])
                color = tuple([np.random.choice([i for i in range(255)]) for i in range(3)])

                self.dict_shapes[str(m)] = (choice, color, x_location)

                dimension = self.windowsize[0]/self.possibilities

                if choice == 'rect':
                    pygame.draw.rect(self.DISPLAYSURF, color, (x_location, y_location, dimension, dimension))

                elif choice == 'circle':
                    # pygame.draw.circle(self.DISPLAYSURF, color, (x_location, y_location+15), 15)
                    pygame.draw.rect(self.DISPLAYSURF, color, (x_location, y_location, 30, 30))

                else:
                    # pygame.draw.polygon(self.DISPLAYSURF, color, x_location, )
                    # pygame.draw.ellipse(self.DISPLAYSURF, color, rect)
                    pass

        else:
            for i, j in self.dict_shapes.items():
                if j[0] == 'rect':
                    pygame.draw.rect(self.DISPLAYSURF, j[1], (j[2], y_location, 30, 30))
                elif j[0] == 'circle':
                    # pygame.draw.circle(self.DISPLAYSURF, j[1], ((j[2], y_location+15)), 15)
                    pygame.draw.rect(self.DISPLAYSURF, j[1], (j[2], y_location, 30, 30))
                else:
                    pass
        return

    def display(self):
        pygame.display.update()
        self.fpsClock.tick(self.FPS)

        for event in pygame.event.get():

            if event.type == QUIT:
                # self.AgentA.model.save('models/AgentA.h5')
                pygame.quit()
                sys.exit()
        return

    def displayObstacles(self, i):
        number_of_obstacles = np.random.choice([i for i in range(3,10)])
        location = (i * 300)/10
        if i == 0:
            self.show_n_shapes(number_of_obstacles, y_location = location, generate_shapes = True)
        else:
            self.show_n_shapes(number_of_obstacles, y_location = location, generate_shapes = False)
        return True

    def evaluate(self, state):
        obs, reward, done, info = '', False, True, 'failure'
        if state in self.shapes_state:
            reward = False
        else:
            reward = True
        return obs, reward, done, info 

    def step(self, state):
        # Clear previous displays


        # decide state of robot
        # State varies from 0 to 9
        location = (state * 300)/10
        # display the robot
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (location, 270, 30, 30))
        self.display()
        # time.sleep()

        obs, reward, done, info = self.evaluate(state)
        return obs, reward, done, info


    def binaries(self):
        binaries = []
        for k in range(10):
            if k in self.shapes_state:
                binaries.append(1)
            else:
                binaries.append(0)

        return binaries


def TestNetwork(iterations = 500):
    # run the newly developed neural network to decide while faced with the obstacles in the environment.
    model = load_model('./models/agent_A.h5')
    robo = RoboObstacle()
    i = 0
    state = 0
    iteration = 0
    successes = 0
    while iteration < iterations:
        # display Obstacles
        robo.DISPLAYSURF.fill(robo.WHITE)
        robo.displayObstacles(i)

        # predict movement
        # binarize Obstacles
        binaries = np.array(robo.binaries()).reshape(1, -1)
        q_value = np.argmax(model.predict(binaries))

        # execute step
        obs, reward, done, info = robo.step(q_value)
        successes+= reward

        i+= 1

        if i == 10:
            i = 0
        else:
            pass

        iteration += 1


        if 0xFF == ord('q'):
            break
    return successes, iterations



def TrainNetwork(iterations = 5000, model_name = 'agent_4'):
    robo = RoboObstacle(storageSize = iterations)
    # print(len(list(robo.Agent.replay_memory)))
    i = 0
    state = 0
    iteration = 0
    successes = 0
    # while iteration < iterations:
    while True:
        # display Obstacles
        robo.DISPLAYSURF.fill(robo.WHITE)
        robo.displayObstacles(i)

        # binarize Obstacles
        binaries = np.array(robo.binaries()).reshape(1, -1)


        # agent make choice
        # choice = np.random.choice([i for i in range(10)])
        q_value = np.argmax(robo.Agent.model.predict(binaries))

        obs, reward, done, info = robo.step(q_value)
        action = robo.Agent.epsilon_greedy(q_value, iteration)
        successes+= reward


        # Let's memorize what just happened
        baseline_value = robo.Agent.replay_memory_size
        # if successes > int(baseline_value/2) and iteration == len(list(robo.Agent.replay_memory)):
        if successes == int(baseline_value) and reward == True:
            robo.Agent.model.save('./models/{}.h5'.format(model_name))
            break
        elif successes < int(baseline_value * 0.8) and reward == True:
            # replace random false with new inputs
            robo.Agent.replay_memory.append((np.array(binaries[0]), action, reward, action, 1.0 - done))
        
        else:
            pass
        

        # robo.Agent.stats()

        # Train Network
        robo.trainDQN()
        i+= 1

        if i == 10:
            i = 0
        else:
            pass

        iteration += 1


        if 0xFF == ord('q'):
            break


    lengthOfQueue = len(list(robo.Agent.replay_memory))
    return successes, iteration, lengthOfQueue



if __name__ == "__main__":
    success, iteration, lengthOfQueue = TrainNetwork()
    print('current iteration: ', str(iteration), ' Total Successes: ', str(success), ' length of queue: ', lengthOfQueue)

    """
    Result:
    current iteration:  4999  Total Successes:  2810  length of queue:  4999
    success:  2810
    """

    # success, iterations = TestNetwork()
    # print('Total Successes: ', str(success), ' Total Iteration: ', str(iterations))

    """
    Result: T
    otal Successes:  290  Total Iteration:  500
    """
          
