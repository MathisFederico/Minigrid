""" Module for FeatureConditions for handcrafted options. """

import numpy as np
from option_graph import FeatureCondition
from PIL import Image

from gym_minigrid.options.utils import check_color, check_object, insert_in_text
from gym_minigrid.minigrid import OBJECT_TO_IDX, DIR_TO_VEC, TILE_PIXELS
from gym_minigrid.minigrid import WorldObj
tuple_to_worldobj = WorldObj.decode


class ObjectFeatureCondition(FeatureCondition):

    """ Base class for object-based FeatureCondition. """

    def __init__(self, obj:str, color:str=None, prefix:str="", suffix:str="", state:int=0) -> None:
        self.obj, self.object_id = check_object(obj)
        self.color, self.color_id = check_color(color)
        feature_name = insert_in_text((self.color, self.obj), prefix, suffix)

        image = None
        if self.obj is not None:
            world_obj = tuple_to_worldobj(self.object_id, self.color_id, state)
            if world_obj is not None:
                image = np.zeros(shape=(TILE_PIXELS * 3, TILE_PIXELS * 3, 4), dtype=np.uint8)
                world_obj.render(image)
                self.obj_image = image

        super().__init__(feature_name, image=image)

    def _is_object(self, observation):
        is_object = observation[..., 0] == self.object_id
        if self.color is not None:
            is_color = observation[..., 1] == self.color_id
            is_object = is_object and is_color
        return is_object

    def __call__(self, observation) -> int:
        raise NotImplementedError


class IsObjectRight(ObjectFeatureCondition):

    """ Is the given object on the right of the agent's view ? """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is", suffix=" on right?")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        return np.any(is_object[is_object.shape[1]//2+1:, :])


class IsObjectLeft(ObjectFeatureCondition):

    """ Is the given object on the left of the agent's view ? """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is", suffix=" on left?")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        return np.any(is_object[:is_object.shape[1]//2, :])


class IsObjectOnSides(ObjectFeatureCondition):

    """ Is the given object on the sides of the agent ? """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is", suffix=" on sides?")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        return np.any(is_object[:, -1])


class IsObjectForward(ObjectFeatureCondition):

    """ Is the given object in front of the agent ? """

    def __init__(self, obj:str, color:str=None, distance:int=1) -> None:
        super().__init__(obj=obj, color=color, prefix="Is", suffix=f" forward ({distance})?")
        self.distance = distance

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        return is_object[3, -(self.distance+1)]


class HoldObject(ObjectFeatureCondition):

    """ Is the given object in the agent's hand ? """

    def __init__(self, obj:str=None, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is agent holding", suffix="?")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.obj is not None:
            is_object = self._is_object(observation)
        else:
            holdable_object = ('key', 'ball', 'box')
            holdable_indexes = [OBJECT_TO_IDX[obj] for obj in holdable_object]
            is_object = np.isin(observation[..., 0], holdable_indexes)
        return is_object[3, -1]


class IsForwardBlocked(FeatureCondition):

    """ Is the front of the agent blocked ? """

    def __init__(self, distance:int=1, exact=True) -> None:
        super().__init__(name="Is forward blocked?")
        self.door_id = 4
        self.can_overlap_blocked_ids = [1, 3, 8, 9, 10]
        self.distance = distance
        self.exact = exact

    def _can_overlap(self, observation):
        is_an_open_door = np.logical_and(observation[..., 0] == 4, observation[..., 2] == 0)
        can_overlap = np.isin(observation[..., 0], self.can_overlap_blocked_ids)
        return np.logical_or(can_overlap, is_an_open_door)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        can_overlap = self._can_overlap(observation)
        if self.exact:
            return not can_overlap[3, -(self.distance + 1)]
        return not np.all(can_overlap[3, -(self.distance + 1):-1])


class IsOnScreen(ObjectFeatureCondition):

    """ Is the given object on screen ? """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is", suffix=" on screen?")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        return np.any(is_object)


class IsDoorLocked(ObjectFeatureCondition):

    """ Is the door locked ?

    This FeatureCondition has MEMORY !

    """

    def __init__(self, color:str=None) -> None:
        super().__init__(obj='door', color=color, state=2, prefix="Is", suffix=" locked?")
        self.locked_doors = {}

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        locked_door_indexes = np.where(np.logical_and(is_object, observation[..., 2] == 2))
        locked_colors = observation[..., 1][locked_door_indexes]
        for color_id in locked_colors:
            self.locked_doors[color_id] = True

        unlocked_door_indexes = np.where(np.logical_and(is_object, observation[..., 2] != 2))
        unlocked_colors = observation[..., 1][unlocked_door_indexes]
        for color_id in unlocked_colors:
            self.locked_doors[color_id] = False

        if self.color is not None:
            return self.locked_doors[self.color_id]
        return any(list(self.locked_doors.values()))


class IsDoorClosed(ObjectFeatureCondition):

    """ Is the door closed ?

    This FeatureCondition has MEMORY !

    """

    def __init__(self, color:str=None) -> None:
        super().__init__(obj='door', color=color, state=1, prefix="Is", suffix=" closed?")
        self.closed_doors = {}

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        is_object = self._is_object(observation)
        closed_door_indexes = np.where(np.logical_and(is_object, observation[..., 2] != 0))
        closed_colors = observation[..., 1][closed_door_indexes]
        for color_id in closed_colors:
            self.closed_doors[color_id] = True

        open_door_indexes = np.where(np.logical_and(is_object, observation[..., 2] == 0))
        open_colors = observation[..., 1][open_door_indexes]
        for color_id in open_colors:
            self.closed_doors[color_id] = False

        if self.color is not None:
            return self.closed_doors[self.color_id]
        return any(list(self.closed_doors.values()))


class IsForwardFloorNotBlockingDoor(FeatureCondition):

    """ Is forward a empty not blocking a door ? """

    def __init__(self) -> None:
        super().__init__(name='Is forward empty not blocking door?')
        self.is_floor = IsObjectForward('empty')

    def __call__(self, observation) -> int:
        is_forward_floor = self.is_floor(observation)
        is_door = observation[..., 0] == OBJECT_TO_IDX['door']
        is_blocking_door = np.any([is_door[3, -1], is_door[4, -2], is_door[2, -2], is_door[3, -3]])
        is_wall = observation[..., 0] == OBJECT_TO_IDX['wall']
        is_blocking_path = np.all([is_wall[2, -1], is_wall[4, -1]])
        return is_forward_floor and not is_blocking_door and not is_blocking_path


class IsDoorBlocked(ObjectFeatureCondition):

    """ Is a door in screen blocked by a given object ?

    This FeatureCondition has MEMORY !

    """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj=obj, color=color, prefix="Is door blocked by", suffix="?")
        self.blocked_doors = {}

    def __call__(self, observation) -> int:
        is_object = self._is_object(observation)

        is_door = observation[..., 0] == OBJECT_TO_IDX['door']
        door_indexes = np.where(is_door)

        for door_coords in np.swapaxes(door_indexes, 0, 1):
            door_color = observation[..., 2][tuple(door_coords)]
            this_door_is_blocked = False
            for direction in DIR_TO_VEC:
                coords = door_coords + direction
                if np.all(np.logical_and(coords >=0, coords < 7)):
                    if is_object[tuple(coords)]:
                        this_door_is_blocked = True
            self.blocked_doors[door_color] = this_door_is_blocked

        return any(list(self.blocked_doors.values()))
