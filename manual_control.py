#!/usr/bin/env python3

import argparse
import gym

from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.options.feature_conditions import *
from gym_minigrid.options.options import *


class ManualControl:

    def __init__(self, env:MiniGridEnv, window:Window, config, features=None, options=None) -> None:
        self.env = env
        self.agent_view = config.agent_view
        self.tile_size = config.tile_size
        self.seed = config.seed
        self.window = window
        self.features = features if features is not None else []
        self.options = options if options is not None else []

    def redraw(self, obs):
        img = obs['image'] if isinstance(obs, dict) else obs
        if not self.agent_view:
            img = self.env.render('rgb_array', tile_size=self.tile_size)
        else:
            img = self.env.get_obs_render(img, tile_size=self.tile_size)
        self.window.show_img(img)

    def reset(self):
        if self.seed != -1:
            self.env.seed(self.seed)
        obs = self.env.reset()
        if hasattr(env, 'mission'):
            print(f'Mission: {env.mission}')
            window.set_caption(env.mission)
        self.redraw(obs)

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        obs = obs['image']

        if len(self.options) > 0:
            print("Options:")
        for option in self.options:
            action_id = option(obs)
            print(f"   {option}: {str(action_id)}")

        if len(self.features) > 0:
            print("Features:")
        for feats in self.features:
            print(f"   {feats}: {feats(obs)}")

        print(f'Step={self.env.step_count}, Reward={reward}')
        if done:
            print('Done!')
            self.reset()
        else:
            self.redraw(obs)

    def manual_policy(self, key:str):
        if key == 'escape':
            return self.window.close()
        if key == 'backspace':
            return self.reset()
        if key == 'left':
            return self.env.actions.left
        if key == 'right':
            return self.env.actions.right
        if key == 'up':
            return self.env.actions.forward
        if key == ' ':
            return self.env.actions.toggle
        if key == 'pageup':
            return self.env.actions.pickup
        if key == 'pagedown':
            return self.env.actions.drop
        if key == 'enter':
            return self.env.actions.done

    def key_handler(self, event):
        print(f'Pressed: {event.key}')
        action = self.manual_policy(event.key)
        self.step(action)

    def run(self):
        self.manual_policy('backspace')
        window.reg_key_handler(self.key_handler)
        window.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )

    config = parser.parse_args()
    env = gym.make(config.env)
    window = Window('gym_minigrid - ' + config.env)

    features = [
    ]

    options = [
        PickupObject('key'),
        PickupObject('box'),
        PickupObject('ball'),
        OpenLockedDoor(),
        OpenBlockedDoor(),
        DropAway(),
    ]

    control = ManualControl(env=env, window=window, config=config, 
        features=features, options=options)
    control.run()
