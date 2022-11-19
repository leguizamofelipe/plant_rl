from time import sleep
from isaacgym_sim import Simulation
import numpy as np

S = Simulation()


while not S.gym.query_viewer_has_closed(S.viewer):
    S.sim_step()
