import math
import numpy as np
import matplotlib.pyplot as plt
from basic_model.point import Point
from basic_model.kinematics import *

class PlantModel:
    def __init__(self, fruit_radius, link_len, n_joints):
        # All lengths in cm
        self.link_length = link_len
        self.total_joints = n_joints
        self.fruit_radius = fruit_radius
        self.max_occlusion = self.link_length * self.total_joints / self.fruit_radius
        # self.dh_table = np.array([
        #         # Assume 5 links
        #         #       a_i-1      |alpha_i-1|    d   | theta
        #         [         0         ,   0    ,    0   ,  0 ], #0
        #         [  self.link_length ,   0    ,    0   ,  0 ], #1
        #         [  self.link_length ,   0    ,    0   ,  0 ], #2
        #         [  self.link_length ,   0    ,    0   ,  0 ], #3
        #         [  self.link_length ,   0    ,    0   ,  0 ], #4
        #         [  self.link_length ,   0    ,    0   ,  0 ], #5
        #     ])
        self.dh_table = np.zeros((self.total_joints+1,4))
        self.dh_table[1:,0] = self.link_length
        self.get_joint_poses()

    def get_joint_poses(self):
        # prev_T = np.eye(4)
        self.pose_list = []
        # Start with a transformation matrix that assumes the plant is shifted to the bottom left (T_u_0)
        prev_T = np.array([[1, 0, 0, -15],
                           [0, 1, 0, -15],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        for joint in range(1, self.total_joints+2):
            prev_T = np.matmul(prev_T, find_T_i(self.dh_table, joint, print_res=False))
            pose = prev_T[0:3][:,3]
            joint_point = Point(pose[0], pose[1], pose[2])
            self.pose_list.append(joint_point)
        return self.pose_list

    def plot_plant(self, points = []):
        self.get_joint_poses()
        ax = plt.axes()
        ax.set_xlim([-20,20]) # Was 0.5
        ax.set_ylim([-20,20]) # Was 0.5
        # ax.set_zlim([0,1])
        x_list = [point.x for point in self.pose_list]
        y_list = [point.y for point in self.pose_list]
        ax.plot(x_list, y_list)
        circle = plt.Circle((0,0), self.fruit_radius, color = 'r')
        ax.add_patch(circle)
        # ax.plot3D(x_list, y_list, z_list, 'blue')
        # ax.set_title(f'Endpoint Pose = {self.endpoint}')
        for point in points:
            ax.scatter([point.x], [point.y])
        plt.tight_layout()
        plt.show()

    def rotate_node(self, node_i, angle):
        # DH table takes angles in degrees
        self.dh_table[int(node_i), 3] = angle
        self.get_joint_poses()

    def find_intersection(self, p_1, p_2):
        d_x = p_2.x-p_1.x
        d_y = p_2.y-p_1.y

        d_r = (d_x**2+d_y**2)**(1/2)

        D = p_1.x*p_2.y-p_2.x*p_1.y
        
        # Discriminant
        delta = self.fruit_radius**2 * d_r**2 - D**2

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

    def find_point_dist(self, p1, p2):      
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

    def find_seg_length(self, seg):
        seg_len = 0
        for count in range(0, len(seg)-1):
            seg_len += self.find_point_dist(seg[count], seg[count+1])
        return seg_len

    def calculate_occlusion(self):
        inscribed_len = 0
        seg = []
        in_circle = False

        for joint in range(0, self.total_joints):
            res = self.find_intersection(self.pose_list[joint], self.pose_list[joint+1])
            # Case where the links enter the circle 
            if len(res) == 1 and len(seg) == 0: 
                seg.append(res[0])
                seg.append(self.pose_list[joint+1])

                in_circle = True
            # Case where the links exit the circle
            elif len(res) == 1 and len(seg) > 0:
                seg.append(res[0])
                inscribed_len += self.find_seg_length(seg)
                seg = []
                in_circle = False
            # Case where link enters and exits the circle
            elif len(res) > 1:
                inscribed_len += self.find_seg_length(seg)
                seg = []
                in_circle = False
            elif in_circle:
                seg.append(self.pose_list[joint+1])
                if joint == self.total_joints-1:
                    inscribed_len += self.find_seg_length(seg)

        return inscribed_len/self.fruit_radius

    def get_angles(self):
        return self.dh_table[0:self.total_joints, 3]