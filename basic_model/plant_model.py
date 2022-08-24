import numpy as np
import matplotlib.pyplot as plt
from basic_model.point import Point
from basic_model.kinematics import *

class PlantModel:
    def __init__(self):
        # All lengths in cm
        self.link_length = 5
        self.total_joints = 5
        self.dh_table = np.array([
                # Assume 5 links
                #       a_i-1      |alpha_i-1|    d   | theta
                [         0         ,   0    ,    0   ,  0 ], #0
                [  self.link_length ,   0    ,    0   ,  0 ], #1
                [  self.link_length ,   0    ,    0   ,  0 ], #2
                [  self.link_length ,   0    ,    0   ,  0 ], #3
                [  self.link_length ,   0    ,    0   ,  0 ], #4
                [  self.link_length ,   0    ,    0   ,  0 ], #5
            ])
        self.get_joint_poses()

    def get_joint_poses(self):
        # prev_T = np.eye(4)
        self.pose_list = []
        # Start with a transformation matrix that assumes the plant is shifted to the bottom left (T_u_0)
        prev_T = np.array([[1, 0, 0, -15],
                           [0, 1, 0, -15],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        for joint in range(1, self.total_joints+1):
            prev_T = np.matmul(prev_T, find_T_i(self.dh_table, joint, print_res=False))
            pose = prev_T[0:3][:,3]
            joint_point = Point(pose[0], pose[1], pose[2])
            self.pose_list.append(joint_point)
        return self.pose_list

    def plot_plant(self, fruit_radius, points):
        self.get_joint_poses()
        ax = plt.axes()
        ax.set_xlim([-20,20]) # Was 0.5
        ax.set_ylim([-20,20]) # Was 0.5
        # ax.set_zlim([0,1])
        x_list = [point.x for point in self.pose_list]
        y_list = [point.y for point in self.pose_list]
        ax.plot(x_list, y_list)
        circle = plt.Circle((0,0), fruit_radius, color = 'r')
        ax.add_patch(circle)
        # ax.plot3D(x_list, y_list, z_list, 'blue')
        # ax.set_title(f'Endpoint Pose = {self.endpoint}')
        for point in points:
            ax.scatter([point.x], [point.y])
        plt.tight_layout()
        plt.show()

    def rotate_node(self, node_i, angle):
        # DH table takes angles in degrees
        self.dh_table[node_i, 3] = angle
        self.get_joint_poses()

    def find_intersection(self, p_1, p_2, fruit_radius):
        d_x = p_2.x-p_1.x
        d_y = p_2.y-p_1.y

        d_r = (d_x**2+d_y**2)**(1/2)

        D = p_1.x*p_2.y-p_2.x*p_1.y
        
        # Discriminant
        delta = fruit_radius**2 * d_r**2 - D**2

        if delta < 0:
            return []
        elif delta == 0:
            x = 1/d_r**2 * D*d_y
            y = 1/d_r**2 * D*d_x
            if x > min(p_1.x, p_2.x) and x < max(p_1.x, p_2.x) and y > min(p_1.y, p_2.y) and y < max(p_1.y, p_2.y):
                return [Point(x, y, 0)]
            else:
                return []
        elif delta > 0:
            res = []

            x_1 = 1/d_r**2 * (D*d_y + np.sign(d_y)*d_x*delta**(1/2))
            y_1 = 1/d_r**2 * (-D*d_x + abs(d_y)*delta**(1/2))
            
            if x_1 > min(p_1.x, p_2.x) and x_1 < max(p_1.x, p_2.x) and y_1 > min(p_1.y, p_2.y) and y_1 < max(p_1.y, p_2.y):
                res = res + [Point(x_1, y_1, 0)]

            x_2 = 1/d_r**2 * (D*d_y - np.sign(d_y)*d_x*delta**(1/2))
            y_2 = 1/d_r**2 * (-D*d_x - abs(d_y)*delta**(1/2))

            if x_2 > min(p_1.x, p_2.x) and x_2 < max(p_1.x, p_2.x) and y_2 > min(p_1.y, p_2.y) and y_2 < max(p_1.y, p_2.y):
                res = res + [Point(x_2, y_2, 0)]

            return res

    def calculate_occlusion(self, fruit_radius):
        res_list = []
        for joint in range(0, self.total_joints-1):
            res = self.find_intersection(self.pose_list[joint], self.pose_list[joint+1], fruit_radius)
            res_list = res_list + res

        return res_list
