import math
import imageio
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time

import os
import numpy as np
import torch

class Simulation():
    def __init__(self) -> None:

        ############################################### SIM SETUP  #############################################
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Set controller parameters
        # IK params
        self.damping = 0.05

        self.device = 'cpu'

        # Parse arguments
        args = gymutil.parse_arguments(description="Franka Attractor Example")

        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2

        args.physics_engine = gymapi.SIM_FLEX

        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
        sim_params.flex.shape_collision_margin = 0.1

        #For deform bodies
        # enable Von-Mises stress visualization
        sim_params.stress_visualization = False
        sim_params.stress_visualization_min = 0.0
        sim_params.stress_visualization_max = 1e6

        sim_params.use_gpu_pipeline = False
        
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        self.img_dir = "interop_images"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
                
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # Set up the env grid
        self.num_envs = 1
        spacing = 3
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        ############################################# FRANKA SETUP #########################################

        # Load franka asset
        asset_root = "./assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        franka_num_dofs = len(franka_dof_props)
        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)
        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]
        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos
        # send to torch
        default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)
        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]
        self.target_angles = np.zeros(9, np.float32)

        ############################################# DEFORMABLE BODY SETUP #########################################

        # Load urdf for deformable body
        soft_asset_file = "urdf/plant.urdf"
        soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)
        asset_soft_body_count = self.gym.get_asset_soft_body_count(soft_asset)
        asset_soft_materials = self.gym.get_asset_soft_materials(soft_asset)
        # Print asset soft material properties
        print('Soft Material Properties:')
        for i in range(asset_soft_body_count):
            mat = asset_soft_materials[i]
            print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

        # Some common handles for later use
        self.envs = []
        self.franka_handles = []
        self.soft_actors = []
        self.cams = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.hand_idxs = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(2, 0.3, -1.5)
            pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, pose, "franka", i, 2)
            self.franka_handles.append(franka_handle)
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # Add deformable body
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 0)
            soft_actor = self.gym.create_actor(env, soft_asset, pose, "soft", i, 1)
            self.soft_actors.append(soft_actor)

            # Add fruit
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(3.2, 0.8, -1.8)
            sphere_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
            ball_actor = self.gym.create_actor(env, sphere_asset, pose, "ball", i, 0, 0)

            # Add base
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            asset_options.fix_base_link = True
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(2, 0.1, -2)
            base_asset = self.gym.create_box(self.sim, 1, 0.1, 1, asset_options)
            base_actor = self.gym.create_actor(env, base_asset, pose, "base", i, 0, 0)

            # add camera
            cam_props = gymapi.CameraProperties()
            cam_props.width = 128
            cam_props.height = 128
            cam_props.enable_tensors = True
            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            self.gym.set_camera_location(cam_handle, env, gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))
            self.cams.append(cam_handle)

            # obtain camera tensor
            raw_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
            self.cam_tensor = gymtorch.wrap_tensor(raw_tensor)
            print("Got camera tensor with shape", self.cam_tensor.shape)

        # point camera at middle env
        cam_pos = gymapi.Vec3(8, 2, 6)
        cam_target = gymapi.Vec3(-8, 0, -6)
        middle_env = self.envs[0]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

        
    ############################################### END CONFIGURE AND INITIALIZE  #############################################
        self.count = 0

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def sim_step(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_particle_state_tensor(self.sim)
        # self.gym.refresh

        state_tensor = torch.tensor(self.target_angles, dtype=torch.float32)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(state_tensor))
        
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        flex_pose_tensor = gymtorch.wrap_tensor(self.gym.acquire_particle_state_tensor(self.sim))
        self.cartesian_pose = flex_pose_tensor[:,0:3].flatten()
        self.robot_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))[:,0] 
        
        self.von_mises = self.find_von_mises()

        fname = os.path.join(self.img_dir, "cam-%04d-%04d.png" % (self.count, 0))
        cam_img = self.cam_tensor.cpu().numpy()
        # imageio.imwrite(fname, cam_img)

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        self.count+=1

    def set_joint_angles(self, angles):

        self.target_angles = angles

    def end_sim(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def find_von_mises(self):
        """Extract the element-wise von Mises stresses."""
        (_, tet_stress) = self.gym.get_sim_tetrahedra(self.sim)               

        # Get von mises stress from Cauchy tensor
        return  [np.sqrt(0.5 * \
                                        ((stress.x.x - stress.y.y) ** 2 \
                                        + (stress.y.y - stress.z.z) ** 2 \
                                        + (stress.z.z - stress.x.x) ** 2 \
                                        + 6 * (stress.y.z ** 2 + stress.z.x ** 2 + stress.x.y ** 2))) for stress in tet_stress]