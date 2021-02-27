import time,os,sys, pygame
import numpy as np
import tensorflow as tf
from pygame.locals import *
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
pygame.init()

class LoadNetwork:
    def __init__(self):
        return
    def load(self):
        networkname = './model/agent.h5'
        return load_model(networkname)


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
        self.shapes_state = []

        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()


    def show_n_shapes(self, n, y_location, generate_shapes = True, xlimit = 10):
        if generate_shapes == True:
            self.dict_shapes = {}
            self.shapes_state = []
            # n varies from 0 to 7
            x_poses = [i for i in range(xlimit)]
            for m in range(n):
                x_position = np.random.choice(x_poses, replace = False)
                self.shapes_state.append(x_position)
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

    def displayObstacles(self, xlimit = 10, ylimit = 10):
        for i in range(ylimit):
            k = np.random.choice([i for i in range(3, xlimit)])
            number_of_obstacles = np.random.choice([i for i in range(k)])
            location = i * 30
            self.show_n_shapes(number_of_obstacles, y_location = location, generate_shapes = True, xlimit = xlimit)
        
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
        time.sleep()

        obs, reward, done, info = self.evaluate(state)
        return obs, reward, done, info




if __name__ == "__main__":
    robo = RoboObstacle()
    robo.DISPLAYSURF.fill(robo.WHITE)
    robo.displayObstacles()
    robo.display()
    i = 0
    state = 0
    iterations = 2000
    iteration = 0
    successes = 0
    while iteration < iterations:
        iteration+= 1
        

        if 0xFF == ord('q'):
            break

        

    print('success: ', successes)