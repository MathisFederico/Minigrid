""" Module for Actions for handcrafted options. """

import os

from option_graph import Action

import gym_minigrid
from gym_minigrid.options.utils import load_image
from gym_minigrid.minigrid import MiniGridEnv
ACTIONS = MiniGridEnv.Actions

images_dir = os.path.dirname(gym_minigrid.__file__)
resources_path = os.path.join(images_dir, 'options', 'images')

class MinigridAction(Action):
    """ Base class for any Minigrid action. """
    def __init__(self, action_name:str) -> None:
        image_path = os.path.join(resources_path, 'actions', f'{action_name}.png')
        image = load_image(image_path)
        super().__init__(getattr(ACTIONS, action_name), name=action_name, image=image)

class Left(MinigridAction):
    """ Press left """
    def __init__(self) -> None:
        super().__init__('left')

class Right(MinigridAction):
    """ Press right """
    def __init__(self) -> None:
        super().__init__('right')

class Forward(MinigridAction):
    """ Press forward """
    def __init__(self) -> None:
        super().__init__('forward')

class Pickup(MinigridAction):
    """ Press pickup """
    def __init__(self) -> None:
        super().__init__('pickup')

class Toggle(MinigridAction):
    """ Press toggle """
    def __init__(self) -> None:
        super().__init__('toggle')

class Drop(MinigridAction):
    """ Press drop """
    def __init__(self) -> None:
        super().__init__('drop')
