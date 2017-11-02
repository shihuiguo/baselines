#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

def test(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy    
    
    saver = tf.train.Saver()
    
    with U.single_threaded_session() as sess:
        
        for mid in range(100):
            U.initialize()    
            saver.restore(sess, 'log/model'+str(100+mid))
            obs=env.reset()        
        
            for _ in range(300):
                action, vpred = pi.act(False, obs)
                obs, r, done, info = env.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])        
                env.render()    
                if done:
                    break
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='HalfCheetah-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    test(args.env, num_timesteps=1e6, seed=args.seed)


if __name__ == '__main__':
    main()
