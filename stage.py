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
        networkname = './model/agent_A.h5'
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
        self.updateBinariesKey = {}
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()


    def show_n_shapes(self, n, y_location, generate_shapes = True, xlimit = 10):
        if generate_shapes == True:
            self.shapes_state = []
            # n varies from 0 to 7
            x_poses = [i for i in range(xlimit)]
            for m in range(n):
                x_position = np.random.choice(x_poses, replace = False)
                self.shapes_state.append(x_position)
                x_location = (x_position * 300)/ 10
                choice = np.random.choice(['rect', 'circle'])
                color = tuple([np.random.choice([i for i in range(255)]) for i in range(3)])

                #shape_state
                shape_key = str(x_location) +'_'+ str(y_location)
                self.dict_shapes[shape_key] = (choice, color, x_location)

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
            number_of_obstacles = 3 #np.random.choice([i for i in range(k)])
            location = i * 30
            self.show_n_shapes(number_of_obstacles, y_location = location, generate_shapes = True, xlimit = xlimit)
        
        return True

    def evaluate(self, state):
        obs, reward, done, info = '', False, True, 'failure'
        if state in self.dict_shapes.keys():
            reward = False
        else:
            reward = True
        return obs, reward, done, info 

    def step(self, state):
        # state is the interpretation of the neural network next coordinate (x_y coordinate location)
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


    def binarize(self, current_coordinate):
        # current_coordinate must follow this format, x_y. 
        # where x is the x coordinate
        # where y is the y coordinate
        # coordinate = str(x) + '_'+ str(y)

        # degree of movement (left, right, front, back, stay_in_position)

        # 10 locations, modify from index, 4 - 10
        # 0 - left
        # 1 - right
        # 2 - Front
        # 3 - Back

        x, y = tuple(int(current_coordinate.split('_')[0]), int(current_coordinate.split('_')[1])))

        self.updateBinariesKey[0] = str(x) + '_'+ str(y - 30)
        self.updateBinariesKey[1] = str(x) + '_'+ str(y + 30)
        self.updateBinariesKey[2] = str(x - 30) + '_'+ str(y)
        self.updateBinariesKey[3] = str(x + 30) + '_'+ str(y)
        for m in range(4, 10):
            self.updateBinariesKey[m] = str(x) + '_'+ str(y) # remain in position


        front_movement = (str(x) + '_'+ str(y - 30)) in self.dict_shapes.keys()
        back_movement = (str(x) + '_'+ str(y + 30)) in self.dict_shapes.keys()
        left_movement = (str(x - 30) + '_'+ str(y)) in self.dict_shapes.keys()
        right_movement = (str(x + 30) + '_'+ str(y)) in self.dict_shapes.keys()

        binary = np.array([front_movement, back_movement, left_movement, right_movement, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        return binary

    
    def interpret(self, prediction):
        # interpret the result: x_y
        try:
            output = self.updateBinariesKey[prediction].split('_')
        except:
            output = self.updateBinariesKey[10].split('_')
        return output






if __name__ == "__main__":
    robo = RoboObstacle()
    robo.DISPLAYSURF.fill(robo.WHITE)
    robo.displayObstacles()
    robo.display()
    i = 0
    state = 0
    iterations = 2000000
    iteration = 0
    successes = 0
    while iteration < iterations:
        iteration+= 1
        print(iteration)
        # predict and execute movement
        

        if 0xFF == ord('q'):
            break

        

    print('success: ', successes)