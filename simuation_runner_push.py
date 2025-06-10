import mujoco
from mujoco import viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from config_loader.config_loader import load_config, load_actor_network
from robot_descriptions import aliengo_mj_description
import pygame

# Import the utility functions
from utils import scale_axis, swap_legs, clip_torques_in_groups, quat_rotate_inverse

import flax.linen as nn
import pickle
import jax
from jax import numpy as jnp
import os

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        # Output initialized to 1 (probability of survival)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.ones)(x)
        # x = nn.Dense(1)(x)
        return x.squeeze(-1)
    
class CriticEvaluator:
    def __init__(self, model, params):
        self.apply_fn = model.apply
        self.params = params

# Setup per la critic network
critic_model = CriticNetwork()
with open('VF_models/VF_safe_MPC_small.pkl', 'rb') as f:
    data = pickle.load(f)
params, mean, std = data['model_params'], data['mean'], data['std']

# Creiamo l'oggetto per eseguire la valutazione della critic network
critic_network = CriticEvaluator(critic_model, params)

def normalize_inputs(obss, mean, std):
    return (obss - mean) / (std + 1e-8)

@jax.jit
def critic_inference(params, obs):
    return critic_model.apply(params, obs)

# Initialize pygame and the joystick module
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected")
else:
    joystick = pygame.joystick.Joystick(0)  # Get the first joystick
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")

# Inclination threshold in degrees
INCLINATION_THRESHOLD = 45  # Degrees

# Main simulation loop
def run_simulation(config_path='config.yaml', decimation=10):
    # Load configuration and actor network
    config = load_config(config_path)
    actor_network = load_actor_network(config)

    # Extract configuration parameters
    timestep = config['simulation']['timestep_simulation']
    #timestep_policy = config['simulation']['timestep_policy']
    default_joint_angles = np.array(config['robot']['default_joint_angles'])
    kp_custom = np.array(config['robot']['kp_custom'])
    kd_custom = np.array(config['robot']['kd_custom'])
    scaling_factors = config['scaling']

    # Friction parameters
    min_mu_v, max_mu_v = config["robot"]["mu_vRange"]
    min_Fs, max_Fs = config["robot"]["FsRange"]
    actions = np.zeros(12)  # Initial actions
    torques = np.zeros(12)  # Initial torques

    # Load the MuJoCo model and data
    xml = "aliengo/aliengo.xml"  # aliengo_mj_description.MJCF_PATH
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Initialize robot's state
    data.qpos = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
    renderer = mujoco.Renderer(model)

    grav_tens = torch.tensor([[0., 0., -1.]], device='cuda:0', dtype=torch.double)

    timecounter = 0
    step_counter = 0
    current_actions = np.copy(actions)

    disable_torques = False  # Flag to permanently disable torques

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if timecounter % 1000 == 0 and timecounter > 10:
                data.qvel[1] += 5.0
                print("-------------------------------- ROBOT PUSHED -------------------------------- ")

            timecounter += 1
            step_start = time.time()

            # Check the robot's inclination
            body_quat = data.qpos[3:7].copy()
            inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2)) * (180 / np.pi)

            if inclination > INCLINATION_THRESHOLD:
                disable_torques = True

            if disable_torques:
                torques = np.zeros_like(data.ctrl)  # Set all torques to zero permanently
            else:
                # Handle joystick input
                if pygame.joystick.get_count() == 1:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            viewer.close()

                    # Get axis values
                    axes = [joystick.get_axis(i) for i in [0, 1, 2, 5]]
                    summed_axes = (1 + axes[2]) - (1 + axes[3])  # Combine triggers
                    axes = np.array([axes[0], axes[1], summed_axes])

                    # Scale and process joystick inputs
                    scaled_axes = [scale_axis(i, axes[i]) for i in range(len(axes))]
                    scaled_axes[0], scaled_axes[1] = scaled_axes[1], scaled_axes[0]
                    commands = np.array(scaled_axes)

                    # Apply deadzone threshold
                    threshold = 0.05
                    commands = np.array([x if abs(x) >= threshold else 0 for x in scaled_axes])
                else:
                    commands = np.zeros(3)
                    commands[0] = 1.0

                joint_angles1 = data.qpos[7:].copy()
                joint_angles = swap_legs(joint_angles1)

                joint_velocities1 = data.qvel[6:].copy()
                joint_velocities = swap_legs(joint_velocities1)

                # Update simulation only every `decimation` steps
                if step_counter % decimation == 0: #((timestep_policy // timestep)*decimation) == 0:

                    body_vel = data.qvel[3:6].copy()
                    body_quat_reordered = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
                    tensor_quat = torch.tensor(body_quat_reordered, device='cuda:0', dtype=torch.double).unsqueeze(0)
                    gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)

                    # Scale input data
                    scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
                    scaled_commands = commands[:2] * scaling_factors['commands']
                    scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
                    scaled_gravity_body = gravity_body[0].cpu() * scaling_factors['gravity_body']
                    scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
                    scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
                    scaled_actions = current_actions * scaling_factors['actions']
                    input_data = np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                                 scaled_joint_angles, scaled_joint_velocities, scaled_actions))
                    obs = torch.tensor(input_data, dtype=torch.float32)
                    obs = actor_network.norm_obs(obs)

                    # Get actions from policy
                    with torch.no_grad():
                        current_actions = actor_network(obs).numpy()

                qDes = 0.5*current_actions + default_joint_angles
                # Compute torques
                torques1 = (kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities)

                # Compute torques
                if not disable_torques:
                    torques = swap_legs(torques1)
                    torques = clip_torques_in_groups(torques)

            # Apply torques and step simulation
            noise_std = 0.0 # 15
            noise = np.random.normal(0, noise_std, size=torques.shape)
            noisy_torques = torques + noise
            
            data.ctrl = noisy_torques
            mujoco.mj_step(model, data)

            # -------------------------------
            # Calculate if safe
            # -------------------------------
            body_quat_reordered = np.array([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
            tensor_quat = torch.tensor(body_quat_reordered, device='cuda:0', dtype=torch.double).unsqueeze(0)
            gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)[0].cpu().numpy()
            body_lin_vel_global = data.qvel[:3].copy()
            body_ang_vel_global = data.qvel[3:6].copy()
            body_lin_vel_tensor = torch.tensor(body_lin_vel_global[None], device='cuda:0', dtype=torch.double)
            body_lin_vel_local = quat_rotate_inverse(tensor_quat, body_lin_vel_tensor)[0].cpu().numpy()
            vel_tp1 = np.concatenate([body_lin_vel_local, body_ang_vel_global])
            joint_pos_tp1 = swap_legs(data.qpos[7:].copy())
            joint_vel_tp1 = swap_legs(data.qvel[6:].copy())
            z_after = data.qpos[2]

            obs_flax_np = np.concatenate((
                np.array([z_after], dtype=np.float32),
                gravity_body.astype(np.float32),
                vel_tp1.astype(np.float32),
                joint_pos_tp1.astype(np.float32),
                joint_vel_tp1.astype(np.float32)
            ))

            obs_flax = jnp.array(obs_flax_np)  # <-- qui è jax.numpy array

            # Normalize the observation
            obs_flax = normalize_inputs(obs_flax, mean, std)

            # V_safe = critic_network.apply_fn(critic_network.params, obs_flax)
            V_safe = critic_inference(critic_network.params, obs_flax)

            if V_safe > 0.3:
                print(f"\033[92mV_safe: {V_safe:.4f}\033[0m")
            else:
                print(f"\033[91mV_safe: {V_safe:.4f}\033[0m")
        
            # Update viewer
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            # Increment step counter
            step_counter += 1

            # Maintain real-time step duration
            time_until_next_step = timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    renderer.close()


if __name__ == '__main__':
    run_simulation()
