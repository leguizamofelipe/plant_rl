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
import cv2

np.set_printoptions(suppress=True)

class Simulation():
    def __init__(self, n_envs = 1) -> None:

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

        self.grayscale_shape = (40, 40)

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
        self.num_envs = n_envs
        spacing = 8
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
        self.franka_upper_limits = franka_upper_limits
        self.franka_lower_limits = franka_lower_limits

        # Adjust the lims a bit to force franka to hit the plant
        # self.franka_lower_limits[0] = 0.65

        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        franka_num_dofs = len(franka_dof_props)
        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(400)
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
        self.red_indexes = np.zeros(self.num_envs)
        self.at_targets = [[False]*9]*self.num_envs

        self.current_angles = [np.zeros(9, np.float32)]*self.num_envs
        self.target_angles = [np.zeros(9, np.float32)]*self.num_envs

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            pose = gymapi.Transform()
            franka_pose = gymapi.Vec3(2, 0.3, -1.2)
            pose.p = franka_pose
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

            deform_position = gymapi.Vec3(1.9, 0, -2.1)

            # Add deformable body
            pose = gymapi.Transform()
            pose.p = deform_position
            soft_actor = self.gym.create_actor(env, soft_asset, pose, "soft", i, 1)
            self.soft_actors.append(soft_actor)

            # Add fruit
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            pose = gymapi.Transform()
            fruit_pose = gymapi.Vec3(2, 0.1, -3)
            pose.p = fruit_pose
            sphere_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
            ball_actor = self.gym.create_actor(env, sphere_asset, pose, "ball", i, 0, 0)
            self.gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*(1, 0, 0)))

            # Add base
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            asset_options.fix_base_link = True
            pose = gymapi.Transform()
            pose.p = deform_position#gymapi.Vec3(2, 0.1, -2)
            pose.p.x+=0.1
            pose.p.z+=0.1
            base_asset = self.gym.create_box(self.sim, 0.3, 0.1, 0.3, asset_options)
            base_actor = self.gym.create_actor(env, base_asset, pose, "base", i, 0, 0)

            cam_pos = franka_pose
            cam_pos.z-=0
            cam_pos.y+=1 # Height
            cam_target = fruit_pose

            # add camera
            cam_props = gymapi.CameraProperties()
            cam_props.width = 128
            cam_props.height = 128
            cam_props.enable_tensors = True
            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            self.gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
            self.cams.append(cam_handle)

            # point camera at env
            self.gym.viewer_camera_look_at(self.viewer, env, cam_pos, cam_target)

            self.gym.prepare_sim(self.sim)
            self.get_franka_angles(i)
        
        n_xyz = gymtorch.wrap_tensor(self.gym.acquire_particle_state_tensor(self.sim)).shape[0]
        self.n_plant_xyz_points = int(n_xyz/self.num_envs)

        self.update_image_tensors()
                    
    ############################################### END CONFIGURE AND INITIALIZE  #############################################
        self.count = 0

    # def control_ik(self, dpose):
    #     # solve damped least squares
    #     j_eef_T = torch.transpose(self.j_eef, 1, 2)
    #     lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
    #     u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
    #     return u

    def sim_step(self, skip_images = False, step_printout = False):
        t_0 = time.time()
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # net_contact_forces = self.gym.get_env_rigid_contact_forces(self.envs[0])
        # print(net_contact_forces)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_particle_state_tensor(self.sim)

        self.top_ten_mean, self.von_mises= self.find_von_mises()

        self.update_plant_pose_tensor()
        
        if not skip_images:
            self.update_image_tensors()
            self.calc_red_index()

        for env_n in range(0, self.num_envs): self.get_franka_angles(env_n) 

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        self.count+=1
        t_1 = time.time()

        if step_printout: print(f'Stepped in {t_1-t_0} s')
        # self.error = self.target_angles - np.array(self.current_angles)
        # return max(abs(self.error)) < 0.02

    def end_sim(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def find_von_mises(self):
        """Extract the element-wise von Mises stresses."""
        (_, tet_stress) = self.gym.get_sim_tetrahedra(self.sim)               

        # Get von mises stress from Cauchy tensor
        von_mises = np.array([np.sqrt(0.5 * \
                                                ((stress.x.x - stress.y.y) ** 2 \
                                                + (stress.y.y - stress.z.z) ** 2 \
                                                + (stress.z.z - stress.x.x) ** 2 \
                                                + 6 * (stress.y.z ** 2 + stress.z.x ** 2 + stress.x.y ** 2))) for stress in tet_stress])
        sorted_index_array = np.argsort(von_mises)
        sorted_array = von_mises[sorted_index_array]
        n = 10
        top_ten = sorted_array[-n : ]

        return top_ten.mean(), von_mises

    def get_franka_angles(self, env_n):
        state = self.gym.get_actor_dof_states(self.envs[env_n], self.franka_handles[env_n], gymapi.STATE_POS)
        self.current_angles[env_n] = np.array(list(zip(*state))[0], dtype = np.float32)

        return self.current_angles[env_n]

    def set_franka_angles(self, angles, env_n, skip_timeout = False):
        self.set_franka_angles_target(angles, env_n)

        # state_tensor = torch.tensor(self.target_angles, dtype=torch.float32)
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(state_tensor))

        # self.sim_step()

        there = False
        t_0 = time.time()
        timeout = 0
        if not skip_timeout:
            while not there and not self.gym.query_viewer_has_closed(self.viewer): 
                self.sim_step(skip_images=True)
                there = timeout>40
                timeout+=1

        t_f = time.time()

        # print(f'\nCompleted move to {self.get_franka_angles(env_n)}')
        # print(f'Target was {angles}')
        print(f'\nCompleted move in {t_f-t_0} s, error was: {180/np.pi*np.array(self.get_franka_angles(env_n)-angles)}')
        
    def set_franka_angles_target(self, angles, env_n):
        angles = np.clip(angles, a_min = self.franka_lower_limits, a_max=self.franka_upper_limits, dtype = np.float32)
        self.target_angles[env_n] = angles
        self.gym.set_actor_dof_position_targets(self.envs[env_n], self.franka_handles[env_n], np.array(angles))

    def update_plant_pose_tensor(self):
        flex_pose_tensor = gymtorch.wrap_tensor(self.gym.acquire_particle_state_tensor(self.sim))
        x_vals = np.split(flex_pose_tensor[:,0], self.num_envs)
        y_vals = np.split(flex_pose_tensor[:,1], self.num_envs)
        z_vals = np.split(flex_pose_tensor[:,2], self.num_envs)

        self.plant_pose = [np.concatenate([x_vals[i], y_vals[i], z_vals[i]]) for i in range(0, self.num_envs)]
        return self.plant_pose

    def update_image_tensors(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # obtain camera tensor
        raw_tensors = [self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.cams[0], gymapi.IMAGE_COLOR) for i in range(0, self.num_envs)]
        self.cam_tensors = [gymtorch.wrap_tensor(tensor) for tensor in raw_tensors]
        self.rgb_cam_imgs = [tensor.cpu().numpy() for tensor in self.cam_tensors]
        self.grayscale_cam_imgs = [self.rgb_to_grayscale(img) for img in self.rgb_cam_imgs]

    def save_image(self, env_n = 0):
        self.update_image_tensors()
        print("Got camera tensor with shape", self.cam_tensors[env_n].shape)
        fname = os.path.join(self.img_dir, "cam-%04d-%04d-%04d.png" % (env_n, self.count, 0))
        imageio.imwrite(fname, self.rgb_cam_imgs[env_n])

    def calc_red_index(self, env_n = 0):
        reds = [img[:,:,0]>210 for img in self.rgb_cam_imgs]
        greens = [img[:,:,1]<40 for img in self.rgb_cam_imgs]
        blues = [img[:,:,2]<40 for img in self.rgb_cam_imgs]

        self.red_indexes = [np.sum(np.logical_and(reds[i],np.logical_and(blues[i], greens[i])))/(reds[i].shape[0]*reds[i].shape[1]) for i in range(0, len(blues))]

    def rgb_to_grayscale(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        gray_rescaled = cv2.resize(gray, self.grayscale_shape, interpolation=cv2.INTER_AREA)
        
        return gray_rescaled