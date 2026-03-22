from __future__ import annotations

import sys
from typing import TYPE_CHECKING

sys.path += ['.', '..']

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import babybench.utils as bb_utils

if TYPE_CHECKING:
    from mimoEnv.babybench import BabyBenchEnv
    from mimoProprioception.proprio import SimpleProprioception

tqdm.format_sizeof = (
    lambda x, divisor=None:
        f'{x:,}'
        if divisor is not None
        else f'{x:5.2f}'
)


def filter_proprio(
        proprio: SimpleProprioception,
        relevant_joints: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    sensors: dict[str, np.ndarray] = proprio.sensor_outputs

    qpos = sensors['qpos']
    qvel = sensors['qvel']
    limits = sensors['limits']

    n_joints = len(proprio.joint_names)
    assert len(qpos) == n_joints
    assert len(qvel) == n_joints
    assert len(limits) == n_joints

    qpos_red = np.array([
        qpos[joint_id]
        for joint_id in relevant_joints
    ])
    qvel_red = np.array([
        qvel[joint_id]
        for joint_id in relevant_joints
    ])

    return qpos_red, qvel_red


def babbling_data(
        config: str,
        anatomy: tuple[str, ...] = ('shoulder', 'elbow', 'hand', 'fingers'),
        touch: bool = False,
        n_steps: int = 100_000,
        data_path: str = '',
        video_path: str = ''
) -> pd.DataFrame:
    with open(config) as f:
        config = yaml.safe_load(f)

    env: BabyBenchEnv = bb_utils.make_env(config)

    act_names = [
        act_name
        for act_idx in range(env.model.nu)
        if (act_name := env.model.actuator(act_idx).name).startswith('act:')
    ]

    joint_names = env.proprioception.joint_names
    rel_joint_ids, rel_joint_names = zip(*[
        (joint_id, joint_name.removeprefix('robot:'))
        for joint_id, joint_name in enumerate(joint_names)
        for part in anatomy
        if part in joint_name
    ])
    joint_attr_names = [
        f'joint:{joint_name}.{attr}'
        for attr in ('pos0', 'dpos', 'pos1')
        for joint_name in rel_joint_names
    ]

    n_actuators = env.model.nu
    n_rel_joints = len(rel_joint_names)

    touch_names = []
    n_touch = 0

    if touch:
        n_touch = env.touch.get_touch_obs().shape[0]
        for touch_id in range(n_touch):
            touch_names += [f'touch:{touch_id}.x', f'touch:{touch_id}.y', f'touch:{touch_id}.z']

    obs, *_ = env.reset()

    data: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    record = bool(video_path)

    env.reset()

    for _ in tqdm(range(n_steps), desc='Step', unit_scale=True):
        qpos0, _ = filter_proprio(env.proprioception, rel_joint_ids)

        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)

        qpos1, _ = filter_proprio(env.proprioception, rel_joint_ids)
        delta_qpos: np.ndarray = qpos1 - qpos0

        data.append(
            np.concatenate(
                (
                    action,
                    qpos0, delta_qpos, qpos1,
                    obs['touch'] if touch else []
                ),
                dtype=np.float32
            )
        )

        if record:
            imgs.append(bb_utils.render(env))

        if terminated or truncated:
            env.reset()

    env.close()

    if record:
        bb_utils.evaluation_video(
            images=imgs,
            save_name=video_path,
            resolution=(480, 480)
        )

    data: np.ndarray = np.array(data)

    n_rows, n_cols = data.shape
    assert n_rows == n_steps
    assert n_cols == n_actuators + n_rel_joints * 3 + n_touch * 3

    df = pd.DataFrame(
        data=data,
        columns=[
            *act_names,
            *joint_attr_names,
            *touch_names
        ],
        dtype=np.float32
    )

    if data_path:
        df.to_pickle(data_path)

    return df


if __name__ == '__main__':
    babbling_data(
        config='configs/config_proprio.yml',
        anatomy=('shoulder', 'elbow', 'hand', 'fingers'),
        touch=False,
        n_steps=1_000_000,
        data_path='data/rnd_proprio_spring-act_steps-1m_fskip-50.pkl'
    )
