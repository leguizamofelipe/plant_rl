"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Attractor
----------------
Positional control of franka panda robot with a target attractor that the robot tries to reach
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Franka Attractor Example")

############################################### BEGIN CONFIGURE AND INITIALIZE  #############################################
args.physics_engine = gymapi.SIM_FLEX

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2

args.physics_engine = gymapi.SIM_FLEX

if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
    sim_params.flex.shape_collision_margin = 0.1

    #For deform bodies
    # enable Von-Mises stress visualization
    sim_params.stress_visualization = True
    sim_params.stress_visualization_min = 0.0
    sim_params.stress_visualization_max = 10e+10


# elif args.physics_engine == gymapi.SIM_PHYSX:
#     sim_params.physx.solver_type = 1
#     sim_params.physx.num_position_iterations = 4
#     sim_params.physx.num_velocity_iterations = 1
#     sim_params.physx.num_threads = args.num_threads
#     sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False


if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "./assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

# Load urdf for deformable body
soft_asset_file = "urdf/plant.urdf"
soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = []
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# Print asset soft material properties
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')


# Set up the env grid
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
franka_handles = []
franka_hand = "panda_hand"

pose = gymapi.Transform()
pose.p = gymapi.Vec3(2, 0.3, -1.5)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create helper geometry used for visualization

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

soft_actors = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2)
    body_dict = gym.get_actor_rigid_body_dict(env, franka_handle)
    props = gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)
    franka_handles.append(franka_handle)

    # Add deformable body
    pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(10, 0, -7)
    pose.p = gymapi.Vec3(0, 0, -4)

    soft_actor = gym.create_actor(env, soft_asset, pose, "soft", i, 1)
    soft_actors.append(soft_actor)

    # Add fruit
    asset_options = gymapi.AssetOptions()
    asset_options.density = 10.0
    # asset_options.color
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(3.2, 0.8, -1.8)
    sphere_asset = gym.create_sphere(sim, 0.1)
    ball_actor = gym.create_actor(env, sphere_asset, pose, "ball", i, 0, 0)


# get joint limits and ranges for Franka
franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)

# override default stiffness and damping values
franka_dof_props['stiffness'].fill(1000.0)
franka_dof_props['damping'].fill(1000.0)

# Give a desired pose for first 2 robot joints to improve stability
franka_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

franka_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props['stiffness'][7:] = 1e10
franka_dof_props['damping'][7:] = 1.0

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

# Time to wait in seconds before moving robot
next_franka_update_time = 1.5

num_dofs = gym.get_asset_dof_count(franka_asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_positions = dof_states['pos']

############################################### END CONFIGURE AND INITIALIZE  #############################################
count = 0

while not gym.query_viewer_has_closed(viewer):
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    for d in range(0, num_dofs):
        dof_positions[d] = 0.01*count
        gym.set_actor_dof_states(env, franka_handle, dof_states, gymapi.STATE_POS)
    
    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    count+=1
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)