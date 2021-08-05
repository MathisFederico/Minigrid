""" Module for handcrafted options using option-graph. """

from typing import Dict
from option_graph import Option, OptionGraph, Action, StochasticAction

from gym_minigrid.options.feature_conditions import *
from gym_minigrid.options.actions import Left, Right, Forward, Pickup, Drop, Toggle
from gym_minigrid.options.utils import check_color, check_object, insert_in_text

class ObjectOption(Option):

    """ Base class for object based Options. """

    def __init__(self, obj:str, color:str=None, prefix:str="", suffix:str="") -> None:
        self.obj, self.object_id = check_object(obj)
        self.color, self.color_id = check_color(color)
        option_id = insert_in_text((self.color, self.obj), prefix, suffix)
        super().__init__(option_id)

    def build_graph(self) -> OptionGraph:
        raise NotImplementedError

class TurnTo(ObjectOption):

    """ Turn the agent towards the given object. """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj, color, prefix="Turn to")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        is_right = IsObjectRight(self.obj, self.color)
        graph.add_edge(is_right, Right(), index=True)

        is_left = IsObjectLeft(self.obj, self.color)
        graph.add_edge(is_right, is_left, index=False)

        graph.add_edge(is_left, Left(), index=True)
        graph.add_edge(is_left, Action(None), index=False)

        return graph

class MoveTo(ObjectOption):

    """ Move the agent towards the given object. """

    def __init__(self, obj:str, color:str=None, distance:int=1) -> None:
        super().__init__(obj, color, prefix="Move to")
        self.distance = distance

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)
        is_on_sides = IsObjectOnSides(self.obj, self.color)
        is_blocked = IsForwardBlocked()
        graph.add_edge(is_on_sides, is_blocked, index=False)
        graph.add_edge(is_on_sides, TurnTo(self.obj, self.color), index=True)

        is_obj_forward = IsObjectForward(self.obj, self.color, distance=self.distance)
        graph.add_edge(is_blocked, Forward(), index=False)
        graph.add_edge(is_blocked, is_obj_forward, index=True)

        graph.add_edge(is_obj_forward, Explore(), index=False)
        graph.add_edge(is_obj_forward, Action(None), index=True)

        return graph


class Explore(Option):

    """ Let the agent explore around. """

    def __init__(self) -> None:
        super().__init__("Explore")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        wall_in_front = IsForwardBlocked(distance=3, exact=False)

        explore_forward = StochasticAction(
            actions=[Left(), Forward(), Right()],
            probs=[0.05, 0.9, 0.05],
            name='explore_forward'
        )
        graph.add_edge(wall_in_front, explore_forward, index=False)

        explore_turn = StochasticAction(
            actions=[Forward(), Right()],
            probs=[0.1, 0.9],
            name='explore_turn_right'
        )
        graph.add_edge(wall_in_front, explore_turn, index=True)

        return graph


class SearchFor(ObjectOption):

    """ Make the agent search for given object. """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj, color, prefix="Search for")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        is_on_screen = IsOnScreen(self.obj, self.color)
        graph.add_edge(is_on_screen, Explore(), index=False)
        graph.add_edge(is_on_screen, Action(None), index=True)

        return graph


class DropAway(ObjectOption):

    """ Make the agent pickup and move away the given object. """

    def __init__(self, obj:str=None, color:str=None) -> None:
        super().__init__(obj, color, prefix="Drop", suffix="away")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        hold = HoldObject(self.obj, self.color)
        is_front_free = IsForwardFloorNotBlockingDoor()
        graph.add_edge(hold, Action(None), index=False)
        graph.add_edge(hold, is_front_free, index=True)

        turn_right = StochasticAction(
            actions=[Forward(), Right()],
            probs=[0.05, 0.95],
            name='turn_right'
        )
        graph.add_edge(is_front_free, turn_right, index=False)
        graph.add_edge(is_front_free, Drop(), index=True)

        return graph


class PickupObject(ObjectOption):

    """ Make the agent pickup the given object. """

    def __init__(self, obj:str, color:str=None) -> None:
        super().__init__(obj, color, prefix="Pickup")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        hold_object = HoldObject()
        is_on_screen = IsOnScreen(self.obj, self.color)
        graph.add_edge(hold_object, is_on_screen, index=False)
        graph.add_edge(hold_object, DropAway(), index=True)

        search_for = SearchFor(self.obj, self.color)
        is_obj_forward = IsObjectForward(self.obj, self.color)
        graph.add_edge(is_on_screen, search_for, index=False)
        graph.add_edge(is_on_screen, is_obj_forward, index=True)

        graph.add_edge(is_obj_forward, MoveTo(self.obj, self.color), index=False)
        graph.add_edge(is_obj_forward, Pickup(), index=True)

        return graph


class OpenDoor(ObjectOption):

    """ Make the agent open the closed but unlocked door. """

    def __init__(self, color:str=None) -> None:
        super().__init__("door", color, prefix="Open")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        is_closed = IsDoorClosed(self.color)
        is_in_front = IsObjectForward(self.obj, self.color)
        graph.add_edge(is_closed, Action(None), index=False)
        graph.add_edge(is_closed, is_in_front, index=True)

        graph.add_edge(is_in_front, Toggle(), index=True)
        graph.add_edge(is_in_front, MoveTo(self.obj, self.color), index=False)

        return graph


class OpenLockedDoor(ObjectOption):

    """ Make the agent open the (possibly) locked door. """

    def __init__(self, color:str=None) -> None:
        super().__init__("door", color, prefix="Unlock and open")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        is_locked = IsDoorLocked(self.color)
        open_door = OpenDoor(self.color)
        hold_key = HoldObject('key', self.color)
        graph.add_edge(is_locked, open_door, index=False)
        graph.add_edge(is_locked, hold_key, index=True)

        pickup_key = PickupObject('key', self.color)
        graph.add_edge(hold_key, pickup_key, index=False)
        graph.add_edge(hold_key, open_door, index=True)

        return graph

class OpenBlockedDoor(ObjectOption):

    """ Make the agent open the (possibly) blocked door. """

    def __init__(self, color:str=None) -> None:
        super().__init__("door", color, prefix="Unblock and open")

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)

        is_locked = IsDoorBlocked('ball')
        graph.add_edge(is_locked, OpenLockedDoor(self.color), index=False)
        graph.add_edge(is_locked, PickupObject('ball'), index=True)

        return graph

def get_all_options() -> Dict[str, Option]:
    all_options = {}
    object_options = [
        MoveTo,
        SearchFor,
    ]
    for obj in ('door', 'key', 'ball', 'box'):
        for option_class in object_options:
            option = option_class(obj)
            all_options[str(option)] = option

    pickupable_options = [
        PickupObject,
        DropAway
    ]
    for obj in ('key', 'ball', 'box'):
        for option_class in pickupable_options:
            option = option_class(obj)
            all_options[str(option)] = option

    doors_options = [
        OpenDoor,
        OpenLockedDoor,
        OpenBlockedDoor,
    ]
    for option_class in doors_options:
        option = option_class()
        all_options[str(option)] = option

    return all_options

ALL_OPTIONS = get_all_options()
