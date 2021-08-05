""" Utilitary functions for handcrafted options. """

from typing import List, Union

from PIL import Image
import numpy as np

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

def check_object(obj:str):
    """ Check if the object is valid and returns it's id.

    Args:
        obj: The object to check.

    Returns:
        (obj, obj_id) where obj is the checked object and obj_id it's id.

    """
    if obj is not None:
        assert obj in OBJECT_TO_IDX, f'Unknowned object {obj} in minigrid'
        return obj, OBJECT_TO_IDX[obj]
    return None, None

def check_color(color:str):
    """ Check if the color is valid and returns it's id.

    Args:
        color: The color to check.

    Returns:
        (color, color_id) where color is the checked color and color_id it's id.

    """
    if color is not None:
        assert color in COLOR_TO_IDX, f'Unknowned color {color} in minigrid'
        return color, COLOR_TO_IDX[color]
    return None, 4

def insert_in_text(elements:List[Union[None,str]],
        prefix:str="", suffix:str="", join_prefix:str=' ') -> str:
    """ Builds a string inserting elements between a prefix and a suffix.

    Args:
        elements: List of elements to insert.
        prefix (Optional): String used as prefix. Default is an empty string.
        suffix (Optional): String used as suffix. Default is an empty string.
        join_prefix (Optional): String used between elements. Default is an single space.

    Returns:
        The resulting string.

    """
    text = prefix
    for arg in elements:
        if arg is not None:
            text += join_prefix + arg
    text += suffix
    return text

def load_image(image_path:str):
    try:
        image = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        image = None
    return np.array(image)
