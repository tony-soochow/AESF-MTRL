from player import Player
from learner import Learner
import gym_dmcontrol
import ray
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg_path = 'cfg/DM_humanoid_Distributed_MTSAC_cfg.json'


is_train = True ############## you need to set

num_cpus = 14 ############## you need to set
num_gpus = 1 ############## you need to set
Player = ray.remote(num_cpus=3, num_gpus=0.15)(Player) ############## you need to set
Learner = ray.remote(num_cpus=2, num_gpus=0.3)(Learner) ############## you need to set
ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
envs = []
skills = ['stand', 'walk']    # humanoid
for s in skills:
    env = gym_dmcontrol.DMControlEnv(domain='humanoid', task=s)
    envs.append(env)

task_distributions = [[0],[1]]


networks = []
if is_train:
    for task_idx_list in task_distributions:
        networks.append(
            Player.remote(
                envs,
                cfg_path, 
                task_idx_list,
                eval_episode_idx=40,
            )
        )
    networks.append(Learner.remote(envs, cfg_path, save_period=20000))
    print('Learner added')
else:
    task_idx_list = [4] ############## you need to set 
    for _ in range(1):
        networks.append(
            Player.remote(
                envs,
                cfg_path, 
                task_idx_list, 
                train_mode=False, 
                write_mode=False,
                render_mode=True,
                trained_model_path="",
                eval_episode_idx=400
            )
        )

ray.get([network.run.remote() for network in networks])
ray.shutdown()
