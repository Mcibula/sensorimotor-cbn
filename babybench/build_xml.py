import json
import sys
from typing import Any

import yaml

sys.path.append('.')
sys.path.append('..')

MODEL = {
    'none': 'v1',
    'self_touch': 'v1',
    'hand_regard': 'v2',
}

SCENE = {
    'base': None,
    'crib': 'crib.xml',
    'cubes': 'cubes.xml',
}

POSITION = {
    'base': 'pos="0 0 .1" euler="0 -90 0"',
    'crib': 'pos="0 0 0.3" euler="0 -90 0"',
    'cubes': 'pos="0 0 0.0467196" quat="0.885453 -0.000184209 -0.464728 -0.000527509"',
}

CONSTRAINTS = {
    'base': '',
    'crib': """
        <equality>
            <weld body1="lower_body"/>
        </equality>
        """,
    'cubes': """
        <equality>
            <weld body1="mimo_location"/>
            <joint joint1="robot:hip_bend1" polycoef="0.532 0 0 0 0"/>
            <joint joint1="robot:hip_lean1" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_rot1" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_bend2" polycoef="0.532 0 0 0 0"/>
            <joint joint1="robot:hip_lean2" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:hip_rot2" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:head_tilt" polycoef="0.4 0 0 0 0"/>
            <joint joint1="robot:head_tilt_side" polycoef="0 0 0 0 0"/>
            <joint joint1="robot:right_hip1" polycoef="-1.22 0 0 0 0"/>
            <joint joint1="robot:right_hip2" polycoef="-0.70 0 0 0 0"/>
            <joint joint1="robot:right_hip3" polycoef="0.538 0 0 0 0"/>
            <joint joint1="robot:right_knee" polycoef="-1.45 0 0 0 0"/>
            <joint joint1="robot:left_hip1" polycoef="-1.41 0 0 0 0"/>
            <joint joint1="robot:left_hip2" polycoef="-0.823 0 0 0 0"/>
            <joint joint1="robot:left_hip3" polycoef="0.612 0 0 0 0"/>
            <joint joint1="robot:left_knee" polycoef="-2.14 0 0 0 0"/>
        </equality>
        """,
}


def build(config: dict[str, Any] | None = None, path_to_assets: str = './MIMo/mimoEnv/assets/') -> str:
    try:
        behavior = config['behavior']
        scene = config['scene']
        actuation_model = config['actuation_model']

    except Exception as e:
        raise ValueError('Missing mandatory options in config file') from e

    # Copy commented config to XML
    xml = '<!--\n'
    xml += json.dumps(config, indent=4)
    xml += '\n-->\n'

    # Add worldbody
    xml += '<mujoco model="MIMo">\n<worldbody>\n'

    # Add MIMo model
    xml += f'<body name="mimo_location" {POSITION[scene]}>\n'
    xml += '<freejoint name="mimo_location"/>\n'
    xml += f'<include file="{path_to_assets}/babybench/mimo_{MODEL[behavior]}.xml"></include>\n'

    # Add BabyBench base
    xml += f'</body>\n</worldbody>\n<include file="{path_to_assets}/babybench/base.xml"></include>\n'

    # Add BabyBench meta
    xml += f'<include file="{path_to_assets}/babybench/meta_{MODEL[behavior]}.xml"></include>\n'

    # Add active actuators
    if actuation_model in ('spring_damper', 'muscle'):
        if config['act_body']:
            xml += f'<include file="{path_to_assets}/babybench/motor_body.xml"></include>\n'
        if config['act_head']:
            xml += f'<include file="{path_to_assets}/babybench/motor_head.xml"></include>\n'
        if config['act_eyes']:
            xml += f'<include file="{path_to_assets}/babybench/motor_eyes.xml"></include>\n'
        if config['act_arms']:
            xml += f'<include file="{path_to_assets}/babybench/motor_arms.xml"></include>\n'
        if config['act_hands']:
            xml += f'<include file="{path_to_assets}/babybench/motor_hands.xml"></include>\n'
        if config['act_fingers']:
            xml += f'<include file="{path_to_assets}/babybench/motor_fingers_{MODEL[behavior]}.xml"></include>\n'
        if config['act_legs']:
            xml += f'<include file="{path_to_assets}/babybench/motor_legs.xml"></include>\n'
        if config['act_feet']:
            xml += f'<include file="{path_to_assets}/babybench/motor_feet_{MODEL[behavior]}.xml"></include>\n'

    elif actuation_model == 'positional':
        if config['act_body']:
            xml += f'<include file="{path_to_assets}/babybench/position_body.xml"></include>\n'
        if config['act_head']:
            xml += f'<include file="{path_to_assets}/babybench/position_head.xml"></include>\n'
        if config['act_eyes']:
            xml += f'<include file="{path_to_assets}/babybench/position_eyes.xml"></include>\n'
        if config['act_arms']:
            xml += f'<include file="{path_to_assets}/babybench/position_arms.xml"></include>\n'
        if config['act_hands']:
            xml += f'<include file="{path_to_assets}/babybench/position_hands.xml"></include>\n'
        if config['act_fingers']:
            xml += f'<include file="{path_to_assets}/babybench/position_fingers_{MODEL[behavior]}.xml"></include>\n'
        if config['act_legs']:
            xml += f'<include file="{path_to_assets}/babybench/position_legs.xml"></include>\n'
        if config['act_feet']:
            xml += f'<include file="{path_to_assets}/babybench/position_feet_{MODEL[behavior]}.xml"></include>\n'

    # Add BabyBench scene
    if scene != 'base':
        xml += f'<include file="{path_to_assets}/babybench/{SCENE[scene]}"></include>\n'

    xml += '</mujoco>'
    return xml


if __name__ == '__main__':
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    xml = build(config)
    print(xml)
