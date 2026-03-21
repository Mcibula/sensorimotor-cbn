import yaml
from tqdm import tqdm

import babybench.eval as bb_eval
import babybench.utils as bb_utils

config = './configs/config_proprio.yml'
n_iter = 1000

with open(config) as f:
    config = yaml.safe_load(f)

env = bb_utils.make_env(config)
evaluation = bb_eval.EVALS[config['behavior']](
    env=env,
    render=True,
    save_dir=config['save_dir']
)

ep = 0
evaluation.reset()
env.reset()

for _ in tqdm(range(n_iter)):
    action = env.action_space.sample()
    *_, terminated, truncated, info = env.step(action)
    evaluation.eval_step(info)

    if terminated or truncated:
        evaluation.end(ep)
        ep += 1

        evaluation.reset()
        env.reset()

evaluation.end(ep)
env.close()
