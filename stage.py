import time,os,sys, pygame
import numpy as np
import tensorflow as tf
from pygame.locals import *
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
from keras.models import load_model
pygame.init()

    


class RoboObstacle:
    def __init__(self, fps=50, dimension = (300, 300)):
        """
        Dimension: must be a factor of 30 for both x and y.
        """
        # set up the window
        self.DISPLAYSURF = pygame.display.set_mode(dimension, 0, 32)
        pygame.display.set_caption('RoboObstacle')
        # set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.dimension = dimension

        self.dict_shapes = {}
        self.shapes_state = []
        self.updateBinariesKey = {}
        self.movement = []
        pygame.init()
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()

    def loadNetwork(self, name = 'agent_A'):
        networkname = './models/{}.h5'.format(name)
        return load_model(networkname)


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
        location_x, location_y = state.split('_')

        # decide state of robot
        # State varies from 0 to 9
        # display the robot
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (int(location_x), int(location_y), 30, 30))
        pygame.draw.rect(self.DISPLAYSURF, self.BLUE, (int(location_x), int(location_y), 25,25))
        pygame.draw.rect(self.DISPLAYSURF, self.RED, (int(location_x), int(location_y), 20, 20))
        pygame.draw.rect(self.DISPLAYSURF, self.WHITE, (int(location_x), int(location_y), 15,15))
        pygame.draw.rect(self.DISPLAYSURF, self.BLUE, (int(location_x), int(location_y), 10, 10))
        pygame.draw.rect(self.DISPLAYSURF, self.GREEN, (int(location_x), int(location_y), 5, 5))
        pygame.draw.rect(self.DISPLAYSURF, self.WHITE, (int(location_x), int(location_y), 3, 3))
        self.display()
        # time.sleep()

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

        x, y = tuple(current_coordinate.split('_'))
        x, y = int(x), int(y)
    

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
        value = np.argmax(prediction)
        output = self.updateBinariesKey[value]
        # except:
        #     output = self.updateBinariesKey[9]
        return output


    def startingCoordinate(self):
        x, y = 0, 0
        m, n = self.dimension
        choice = np.random.choice([i for i in range(m)])
        for i in range(int(choice/30), int(m/30)):
            for j in range(int(n/30)):
                key = str(int(i*30)) + '_' + str(int(j*30))
                if key in self.dict_shapes.keys():
                    pass
                else:
                    x = int(i*30)
                    y = int(j*30)
                    break

            break
        return str(x) + '_' + str(y), (x, y) #key, coordinate


    def TrackNextMove(self, move):
        self.movement.append(move)
        output = False
        counts = np.unique(self.movement, return_counts = True)
        if counts[0][0] == self.updateBinariesKey[9] and counts[1][0] > 2000:
            output = True
        else:
            output = False
        return output






if __name__ == "__main__":
    robo = RoboObstacle()
    model = robo.loadNetwork()
    robo.DISPLAYSURF.fill(robo.WHITE)
    robo.displayObstacles()
    robo.display()
    i = 0
    state = 0
    iterations = 2000000
    iteration = 0
    successes = 0
    key, coordinate = robo.startingCoordinate()
    while iteration < iterations:
        # Make move
        robo.step(key)
        iteration+= 1
        # predict and execute movement
        binaries = np.array(robo.binarize(key)).reshape(1, -1)
        key = robo.interpret(model.predict(binaries))
        # Track next move, if next move is static for 2000 iterations break
        if robo.TrackNextMove(key):
            print('exiting program')
            break
        
        if 0xFF == ord('q'):
            break

        

    print('success: ', successes)