"""
MIMo environment utilities
"""

import re
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import xmltodict


def traverse_xmldict(x: dict[str, Any] | list[Any]) -> Iterator[Any]:
    if isinstance(x, list):
        for item in x:
            if isinstance(item, dict):
                yield from traverse_xmldict(item)

    elif isinstance(x, dict):
        is_leaf = True

        for key, value in x.items():
            if isinstance(value, (dict, list)):
                is_leaf = False

            if key == 'joint':
                if isinstance(value, dict):
                    yield value
                elif isinstance(value, list):
                    yield from traverse_xmldict(value)

            elif key == 'body':
                yield from traverse_xmldict(value)

        if is_leaf:
            yield x


def actuator_info(config_path: str) -> dict[
    Literal['actuators', 'act_joints', 'act_limits'],
    list[str] | dict[str, str | tuple[float, float]]
]:
    with open(config_path, 'rb') as f:
        motors = xmltodict.parse(f)['mujoco']['actuator']['motor']

    actuators = [
        motor['@name']
        for motor in motors
    ]

    act_joints = {
        act: f'joint:{motor["@joint"].removeprefix("robot:")}'
        for motor in motors
        if (act := motor['@name']) in actuators
    }

    act_limits = {
        act: tuple(
            map(
                lambda force: float(force),
                motor['@forcerange'].split(' ')
            )
        )
        for motor in motors
        if (act := motor['@name']) in actuators
    }

    return {
        'actuators': actuators,
        'act_joints': act_joints,
        'act_limits': act_limits
    }


def joint_info(config_path: str) -> dict[
    Literal['joints', 'joint_limits', 'dpos_limits'],
    list[str] | dict[str, tuple[float, float]]
]:
    with open(config_path, 'rb') as f:
        mimo_model = xmltodict.parse(f)['mujoco']['body']

    joint_data = list(traverse_xmldict(mimo_model))

    joint_limits = {
        f'joint:{joint["@name"].removeprefix("robot:")}': tuple(
            map(
                lambda angle: np.deg2rad(float(angle)),
                joint['@range'].split(' ')
            )
        )
        for joint in joint_data
    }

    dpos_limits = {
        f'{joint}.dpos': (q_min - q_max, q_max - q_min)
        for joint, (q_min, q_max) in joint_limits.items()
    }

    joints = list(joint_limits.keys())

    return {
        'joints': joints,
        'joint_limits': joint_limits,
        'dpos_limits': dpos_limits
    }


def variable_list(data_path: str) -> list[str]:
    df: pd.DataFrame = pd.read_pickle(data_path)

    return df.columns.tolist()


def variable_filter(var_list: list[str], pattern: str) -> list[str]:
    re_pattern = re.compile(pattern)

    return [
        var_name
        for var_name in var_list
        if re_pattern.match(var_name) is not None
    ]

