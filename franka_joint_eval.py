from time import sleep
from isaacgym_sim import Simulation
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt
import time

S = Simulation()
poses = [[] for i in range(0, 9)]

count = 0
# S.set_franka_angles(np.array([-0.2, 0.5, 0.75, -2, 1.25, 2.25, -1, 0, 0]), 0, skip_timeout=True)

print(S.franka_lower_limits)
print(S.franka_upper_limits)
t_0 = time.time()

maxes = []

while not S.gym.query_viewer_has_closed(S.viewer):
    S.sim_step()
    print(S.red_indexes)

    # print(max(S.von_mises))

    sorted_index_array = np.argsort(S.von_mises)
    sorted_array = S.von_mises[sorted_index_array]
    n = 10
    rslt = sorted_array[-n : ]

    maxes.append(rslt.mean())

    pose = S.get_franka_angles(0)
    for i in range(0, 9):
        poses[i].append(pose[i])
    count+=1
    # if count>5000: break

t_f = time.time()

fig, ax = plt.subplots(9, sharex=True)
fig.suptitle(f"Transient response over {round(t_f-t_0,2)} seconds")
fig.set_size_inches(10, 10)


for i, pose in enumerate(poses):
    ax[i].axhline(y = S.target_angles[0][i]*180/np.pi, color = 'r', linestyle = '-', alpha=0.7)
    ax[i].axhline(y = S.target_angles[0][i]*180/np.pi+2, color = 'r', linestyle = '--', alpha=0.3)
    ax[i].axhline(y = S.target_angles[0][i]*180/np.pi-2, color = 'r', linestyle = '--', alpha=0.3)
    ax[i].plot(np.array(pose)*180/np.pi, linewidth=3)
    ax[i].grid(b=True, linewidth=0.75, alpha = 0.7)

plt.tight_layout()
plt.show()
