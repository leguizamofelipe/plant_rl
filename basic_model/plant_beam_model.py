import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class PlantBeamModel():
    def __init__(self, P = 0, x_app = 0.5, fruit_radius = 0.5):
        # P = Force applied (N)
        # x_app = Position of force application measured from root (m)
        
        self.resolution = 100

        self.p_len = 3

        self.fruit_radius = fruit_radius
        self.fruit_y_center = 0
        self.fruit_x_center = 2

        # Axis declarations
        self.x = np.linspace(0, self.p_len, num=self.resolution)
        
        # Characteristics of plant cross section
        self.R = 0.015 # Radius (m)
        self.c = 0.015 # Radial distance at which to evaluate bending stress
        self.I = 1/4 * 3.14159 * self.R**4

        # Young's modulus
        self.E = 10e9#10e9 # Al-Zube et al

        # Maximum possible first moment of area
        self.Q = 1/12 * (2*self.R)**3

        self.apply_force(P, x_app)

    def apply_force(self, P, x_app):
        if x_app > self.p_len:
            x_app = self.p_len
        elif x_app < 0:
            x_app = 0.15

        # Bending moment
        self.M = P*x_app - P*self.x
        
        if np.sign(P) == 1:
            self.M[self.M<0] = 0
        elif np.sign(P) == -1:
            self.M[self.M>0] = 0

        # Shear stress
        self.V = -P * np.ones(len(self.x))
        self.V[self.x>x_app] = 0

        # Stress caused by bending moment
        self.sigma = self.M*self.c/self.I

        # Max possible shear stress (neutral surface)
        self.tau = self.Q*self.V/(self.I*self.R)

        self.y = np.zeros(len(self.x))
        self.y[self.x<=x_app] = np.array(P*self.x**2/(6*self.E*self.I)*(3*x_app-self.x))[self.x<=x_app]
        self.y[self.x>x_app] = np.array(P*x_app**2/(6*self.E*self.I)*(3*self.x-x_app))[self.x>x_app]

        self.max_von_mises = (self.sigma**2 + 3*self.tau**2)**(1/2)

    def plot_plant(self, save = False, filename = f'{int(time.time())}.png', title =''):
        ax = plt.axes()
        
        circle = plt.Circle((self.fruit_y_center,self.fruit_x_center), self.fruit_radius, color = 'r')
        ax.add_patch(circle)

        ax.plot(self.y, self.x, color = 'green', linewidth = 10)
        ax.set_xlim([-self.p_len+1, self.p_len-1]) # Was 0.5
        ax.set_ylim([-1, self.p_len+1]) # Was 0.5
        plt.title(title)

        if save:
            # plt.axis('off')
            plt.savefig(filename, dpi=500)
            plt.close()
        else:
            plt.show()

    def calculate_occlusion(self):
        inscribed = 0
        for count, x in enumerate(self.x):
            y = self.y[count]
            if ((y-self.fruit_y_center)**2 + (x-self.fruit_x_center)**2) < self.fruit_radius**2:
                inscribed+=1
        return inscribed/len(self.x)

# plant = PlantBeamModel()
# alphas = []
# for force in range(0, 300):
#     plant.apply_force(force, 1.5)
#     # plant.plot_plant()
#     # plant.calculate_occlusion()
#     k = sum(abs(plant.max_von_mises))*10**-8
#     alpha = -k**2

#     print(max(plant.max_von_mises)*10**-6)
#     alphas.append(alpha)

# plt.plot(range(0,300), alphas)
# plt.show()
# # plant.plot_plant(False)
# print('done')