from time import sleep
from isaacgym_sim import Simulation
import numpy as np

S = Simulation()


while not S.gym.query_viewer_has_closed(S.viewer):
    if S.count < 50:
        S.set_joint_angles(np.ones(9, dtype=np.float32))
    S.sim_step()

    print(S.target_angles)
    print(S.count)

    if S.count>50:
        S.set_joint_angles(np.zeros(9))

