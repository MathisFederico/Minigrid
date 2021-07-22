from typing import List, Union
import numpy as np

from gym_minigrid.minigrid import OBJECT_TO_IDX, STATE_TO_IDX, COLOR_TO_IDX


def check_object(object:str):
    if object is not None:
        assert object in OBJECT_TO_IDX, f'Unknowned object {object} in minigrid'
        return object, OBJECT_TO_IDX[object] 
    else:
        return None, None

def check_color(color:str):
    if color is not None:
        assert color in COLOR_TO_IDX, f'Unknowned color {color} in minigrid'
        return color, COLOR_TO_IDX[color]
    else:
        return None, None

def insert_in_text(elements:List[Union[None,str]], prefix:str="", suffix:str="",
        join_prefix:str=' '):
    text = prefix
    for arg in elements:
        if arg is not None:
            text += join_prefix + arg
    text += suffix
    return text

class IsObjectRight(FeatureCondition):

    def __init__(self, object:str, color:str=None) -> None:
        self.object, self.object_id = check_object(object)
        self.color, self.color_id = check_color(color)
        feature_id = insert_in_text((self.color, self.object), "Is", "on right ?")
        super().__init__(feature_id)

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        is_object = observations[..., 0] == self.object_id

        if self.color is not None:
            is_color = observations[..., 1] == self.color_id
            is_object = is_object and is_color

        return np.any(is_object[:, :is_object.shape[1]//2, :])


class TurnToObject(Option):

    def __init__(self, object:str, color:str=None) -> None:
        self.object, self.object_id = check_object(object)
        self.color, self.color_id = check_color(color)
        option_id = insert_in_text((self.color, self.object), "Move to")
        super().__init__(option_id)

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph()
        return 

# class MoveTo(Option):

#     def __init__(self, object:str) -> None:
#         super().__init__(f"Move to {object}")
#         self.object = object
    
#     def __call__(self, observations, greedy: bool=False):
#         raise NotImplementedError

#     def build_graph(self) -> OptionGraph:
#         graph = OptionGraph()
#         is_on_screen = f"Is {self.object} on perpendicular ?"
#         graph.add_node_feature_condition(is_on_screen)

#         is_in_front = f""


