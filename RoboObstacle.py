import time,os,sys, pygame
import numpy as np
import tensorflow as tf
from pygame.locals import *
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
pygame.init()



class DQN:
    def __init__(self):
        self.learning_rate = 0.001
        self.momentum = 0.95
        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 2000000
        self.replay_memory_size = 500
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
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

        self.model = self.DQNmodel()

        
        return

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(1,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
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
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

    def epsilon_greedy(self, q_values, step):
        self.epsilon = max(self.eps_min, self.eps_max -
                           (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < self.epsilon:
            return np.random.randint(10)  # random action
        else:
            return np.argmax(q_values)  # optimal action



class RoboObstacle:
    def __init__(self, fps=50):
        # set up the window
        self.DISPLAYSURF = pygame.display.set_mode((300, 300), 0, 32)
        pygame.display.set_caption('RoboObstacle')
        # set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.dict_shapes = {}

        self.Agent = DQN()
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()

    def show_n_shapes(self, n, y_location, generate_shapes = True):
        if generate_shapes == True:
            self.dict_shapes = {}
            # n varies from 0 to 7
            x_poses = [i for i in range(10)]
            for m in range(n):
                x_position = np.random.choice(x_poses, replace = False)
                x_location = (x_position * 300)/ 10
                choice = np.random.choice(['rect', 'circle'])
                color = tuple([np.random.choice([i for i in range(255)]) for i in range(3)])

                self.dict_shapes[str(m)] = (choice, color, x_location)

                if choice == 'rect':
                    pygame.draw.rect(self.DISPLAYSURF, color, (x_location, y_location, 30, 30))

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

    def step(self, state):
        # Clear previous displays


        # decide state of robot
        # State varies from 0 to 9
        location = (state * 300)/10
        # display the robot
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (location, 270, 30, 30))
        self.display()
        time.sleep(0.5)
        return



if __name__ == "__main__":
    robo = RoboObstacle()
    i = 0
    while True:
        # display Obstacles
        robo.DISPLAYSURF.fill(robo.WHITE)
        robo.displayObstacles(i)

        # agent make choice
        choice = np.random.choice([i for i in range(10)])
        robo.step(choice)

        i+= 1

        if i == 10:
            i = 0
        else:
            pass


        if 0xFF == ord('q'):
            break
