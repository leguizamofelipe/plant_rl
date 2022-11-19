import math
import imageio
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import os
import numpy as np

class Simulation():
    def __init__(self) -> None:
        # Initialize gym
        self.gym = gymapi.acquire_gym()

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
        
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        self.img_dir = "interop_images"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        
        # self.gym.prepare_sim(self.sim)
        
        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # Load franka asset
        asset_root = "./assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01

        print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
        franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options)

        # Load urdf for deformable body
        soft_asset_file = "urdf/plant.urdf"
        soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

        asset_options = []
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
        cams = []

        for i in range(num_envs):
            # create env
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(self.env)

            # add franka
            self.franka_handle = self.gym.create_actor(self.env, franka_asset, pose, "franka", i, 2)
            body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.franka_handle)
            props = self.gym.get_actor_rigid_body_states(self.env, self.franka_handle, gymapi.STATE_POS)
            hand_handle = body = self.gym.find_actor_rigid_body_handle(self.env, self.franka_handle, franka_hand)
            franka_handles.append(self.franka_handle)

            # Add deformable body
            pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(10, 0, -7)
            pose.p = gymapi.Vec3(0, 0, 0)

            soft_actor = self.gym.create_actor(self.env, soft_asset, pose, "soft", i, 1)
            soft_actors.append(soft_actor)

            # Add fruit
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            # asset_options.color
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(3.2, 0.8, -1.8)
            sphere_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
            ball_actor = self.gym.create_actor(self.env, sphere_asset, pose, "ball", i, 0, 0)

            # Add base
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            asset_options.fix_base_link = True
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(2, 0.1, -2)
            base_asset = self.gym.create_box(self.sim, 1, 0.1, 1, asset_options)
            base_actor = self.gym.create_actor(self.env, base_asset, pose, "base", i, 0, 0)

            # add camera
            cam_props = gymapi.CameraProperties()
            cam_props.width = 128
            cam_props.height = 128
            cam_props.enable_tensors = True
            cam_handle = self.gym.create_camera_sensor(self.env, cam_props)
            self.gym.set_camera_location(cam_handle, self.env, gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))
            cams.append(cam_handle)

            # obtain camera tensor
            raw_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, cam_handle, gymapi.IMAGE_COLOR)
            self.cam_tensor = gymtorch.wrap_tensor(raw_tensor)
            print("Got camera tensor with shape", self.cam_tensor.shape)

        # point camera at middle env
        cam_pos = gymapi.Vec3(8, 2, 6)
        cam_target = gymapi.Vec3(-8, 0, -6)
        middle_env = envs[0]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        # get joint limits and ranges for Franka
        franka_dof_props = self.gym.get_actor_dof_properties(envs[0], franka_handles[0])
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
            self.gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

        # Time to wait in seconds before moving robot
        next_franka_update_time = 1.5

        num_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    ############################################### END CONFIGURE AND INITIALIZE  #############################################
        self.count = 0

    def sim_step(self):

        # Every 0.01 seconds the pose of the attactor is updated
        t = self.gym.get_sim_time(self.sim)

        self.gym.set_actor_dof_states(self.env, self.franka_handle, self.dof_states, gymapi.STATE_POS)
        
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        fname = os.path.join(self.img_dir, "cam-%04d-%04d.png" % (self.count, 0))
        cam_img = self.cam_tensor.cpu().numpy()
        # imageio.imwrite(fname, cam_img)

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        self.count+=1

    def set_joint_angles(self, angles):

        self.dof_states['pos'] = angles
        self.sim_step()

    def end_sim(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)