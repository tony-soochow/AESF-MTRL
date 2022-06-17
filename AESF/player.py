import torch
import numpy as np
from model import Actor
import random
import time

from utils import cfg_read

import redis
import _pickle


class Player():
    def __init__(self, 
            envs,
            cfg_path,
            task_idx_list,
            train_mode=True,
            trained_model_path=None,
            render_mode=False,
            write_mode=True,
            eval_episode_idx=100
        ):

        self.envs_dict = None
        self.env = None
        self.task_inital_state_dict = None
        self.reward_episode = [0]*5

        self.envs = envs
        self.task_idx_list = task_idx_list
        self.train_mode = train_mode
        self.render_mode = render_mode
        self.eval_episode_idx = eval_episode_idx

        self.cfg = cfg_read(path=cfg_path)
        self.set_cfg_parameters()

        self.write_mode = write_mode

        self.server = redis.StrictRedis(host='localhost', password='5241590000000000')
        for key in self.server.scan_iter():
            self.server.delete(key)

        self.build_model()
        self.to_device()

        if trained_model_path != None:
            self.load_model(trained_model_path)
            self.random_step = -1

        if self.train_mode is False:
            assert trained_model_path is not None, \
                'Since train mode is False, trained actor path is needed.'
            self.load_model(trained_model_path)

    def set_cfg_parameters(self):
        self.update_iteration = -2

        self.device = torch.device(self.cfg['device'])
        self.reward_scale = self.cfg['reward_scale']
        self.random_step = int(self.cfg['random_step']) if self.train_mode else 0  
        self.print_period = int(self.cfg['print_period_player'])   
        self.num_tasks = int(self.cfg['num_tasks'])
        self.max_episode_time = self.cfg['max_episode_time']
        self.state_dim = self.cfg['actor']['state_dim']


    def build_model(self):
        self.actor = Actor(self.cfg['actor'], num_tasks=self.num_tasks)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        print('########### Trained model loaded ###########')

    def to_device(self):        
        self.actor.to(self.device)

    def pull_parameters(self):
        parameters = self.server.get('parameters')
        update_iteration = self.server.get('update_iteration')

        if parameters is not None:
            if update_iteration is not None:
                update_iteration = _pickle.loads(update_iteration)
            if self.update_iteration != update_iteration:
                parameters = _pickle.loads(parameters)
                self.actor.load_state_dict(parameters['actor'])
                self.update_iteration = update_iteration
    
    def set_env(self, task_idx):
        self.env = self.envs[task_idx]

    def taskIdx2oneHot(self, taskIdx):
        one_hot = np.zeros((self.num_tasks,))
        one_hot[taskIdx] = 1.
        return one_hot

    def state2mtobs(self, state, taskIdx):
        '''
        input :
            state = (state_dim)
            taskIdx : int
        output :
            mtobs = (state_dim+num_tasks)
        '''
        one_hot = self.taskIdx2oneHot(taskIdx)
        mtobs = np.concatenate((state, one_hot), axis=0)
        return mtobs
    
    def run_episode_once(self, task_idx):
        '''
        Args:
            task_idx (int) : Index of the task which will be run once
        Output:
            episode_reward (float) : It is return of Episode
            delta_total_step (int)
            t (int) : consumed time horizon            
        '''
        self.set_env(task_idx)

        state = self.env.reset()
        state = np.concatenate([state, np.zeros(self.state_dim - state.shape[0])])
        mtobs = self.state2mtobs(state, taskIdx=task_idx)
        episode_reward = 0.
        delta_total_step = -self.task_total_step_dict[task_idx]

        for t in range(1, self.max_episode_time+1):   
            if self.task_total_step_dict[task_idx] < self.random_step:
                action = self.env.action_space.sample()
            else:         
                with torch.no_grad():    
                    mtobs_tensor = torch.tensor([mtobs]).to(self.device).float()
                    action = self.actor.get_action(
                        mtobss=mtobs_tensor, 
                        stochastic=self.train_mode
                    )

            next_state, reward, done, info = self.env.step(action)
            next_state = np.concatenate([next_state, np.zeros(self.state_dim - next_state.shape[0])])
            next_mtobs = self.state2mtobs(next_state, taskIdx=task_idx)
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            masked_done = False if t == self.max_episode_time else done

            if self.train_mode:
                sample = (
                    task_idx,
                    mtobs.copy(), 
                    action.copy(), 
                    self.reward_scale * reward, 
                    next_mtobs.copy(), 
                    masked_done
                )
                self.server.rpush('sample', _pickle.dumps(sample))

                self.pull_parameters()
            else:
                if self.render_mode:
                    self.env.render()
                    time.sleep(0.04)
                    if bool(info['success']) or t > 200:
                        time.sleep(0.5)
                        break
                    
            mtobs = next_mtobs

            self.task_total_step_dict[task_idx] += 1

            if done:
                break

        delta_total_step += self.task_total_step_dict[task_idx]

        return episode_reward, delta_total_step, t

    def run(self):
        total_step = 0
        self.task_total_step_dict = {task_idx : 0 for task_idx in self.task_idx_list}
        episode_idx = 0
        episode_rewards_dict = {task_idx : 0. for task_idx in self.task_idx_list}
        ts_dict = {task_idx : 0. for task_idx in self.task_idx_list}

        # initial parameter copy
        self.pull_parameters()

        while True:
            if self.train_mode:
                for task_idx in self.task_idx_list:
                    episode_reward, delta_total_step, t = self.run_episode_once(task_idx)
                    episode_rewards_dict[task_idx] += episode_reward
                    total_step += delta_total_step
                    ts_dict[task_idx] += t

                episode_idx += 1

                if episode_idx % self.print_period == 0:
                    for task_idx in self.task_idx_list:
                        content = '[Player] Tot_step: {0:<6} \t | Episode: {1:<4} \t | Time: {2:5.2f} \t | Task: {3:<2} \t | Reward : {4:5.3f}'.format(
                            total_step,
                            episode_idx + 1,
                            ts_dict[task_idx]/self.print_period,
                            task_idx,
                            episode_rewards_dict[task_idx]/self.print_period
                        )
                        print(content)
                    if self.write_mode:
                        for task_idx in self.task_idx_list:
                            reward_logs_data = (
                                task_idx,
                                self.task_total_step_dict[task_idx],
                                episode_rewards_dict[task_idx]/self.print_period
                            )
                            self.server.rpush('reward_logs', _pickle.dumps(reward_logs_data))

                    episode_rewards_dict = {task_idx : 0. for task_idx in self.task_idx_list}
                    ts_dict = {task_idx : 0. for task_idx in self.task_idx_list}
            else:
                env_names = ['stand', 'walk']
                for task_idx in self.task_idx_list:
                    reward_task = []
                    for i in range(100):
                        episode_reward, _, _ = self.run_episode_once(task_idx)
                        reward_task.append(episode_reward)
                    print("Env ", env_names[task_idx], " Average reward over 100 episodes: ", np.mean(reward_task), np.std(reward_task))
