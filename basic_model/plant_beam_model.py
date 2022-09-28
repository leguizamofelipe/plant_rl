import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PlantBeamModel():
    def __init__(self, P = 0, x_app = 0.5, fruit_radius = 0.5):
        # P = Force applied (N)
        # x_app = Position of force application measured from root (m)
        
        resolution = 100

        self.p_len = 3

        self.fruit_radius = fruit_radius
        self.fruit_x_pos = 0
        self.fruit_y_pos = 2

        # Axis declarations
        self.x = np.linspace(0, self.p_len, num=resolution)
        
        # Characteristics of plant cross section
        self.R = 0.015 # Radius (m)
        self.c = 0.015 # Radial distance at which to evaluate bending stress
        self.I = 1/4 * 3.14159 * self.R**4

        # Young's modulus
        self.E = 10e9 # Al-Zube et al

        # Maximum possible first moment of area
        self.Q = 1/12 * (2*self.R)**3

        self.apply_force(P, x_app)

    def apply_force(self, P, x_app):
        # Bending moment
        self.M = P*x_app - P*self.x
        self.M[self.M<0] = 0

        # Shear stress
        self.V = -P * np.ones(len(self.x))
        self.V[self.x>x_app] = 0

        # Stress caused by bending moment
        self.sigma = self.M*self.c/self.I

        # Max possible shear stress (neutral surface)
        self.tau = self.Q*self.V/(self.I*self.R)

        self.delta = np.zeros(len(self.x))
        self.delta[self.x<=x_app] = np.array(P*self.x**2/(6*self.E*self.I)*(3*x_app-self.x))[self.x<=x_app]
        self.delta[self.x>x_app] = np.array(P*self.x**2/(6*self.E*self.I)*(3*self.x-x_app))[self.x>x_app]

        self.max_von_mises = (self.sigma**2 + 3*self.tau**2)**1/2

    def plot_plant(self):
        ax = plt.axes()
        
        circle = plt.Circle((self.fruit_x_pos,self.fruit_y_pos), self.fruit_radius, color = 'r')
        ax.add_patch(circle)

        ax.plot(self.delta, self.x)
        ax.set_xlim([-self.p_len+1, self.p_len-1]) # Was 0.5
        ax.set_ylim([-1, self.p_len+1]) # Was 0.5
        plt.show()



    def return_occlusion(self):
        inscribed = 0
        for count, y_pos in enumerate(self.x):
            x_pos = self.delta[count]
            if ((x_pos-self.fruit_x_pos)**2 + (y_pos-self.fruit_y_pos)**2) < self.fruit_radius**2:
                inscribed+=1
        return inscribed/len(self.x)

P = PlantBeamModel(P = 10, x_app=1)
P.plot_plant()
