from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from luxai_s3.wrappers import LuxAIS3GymEnv
from kits.python.agent import Agent
from rl.wrappers import RLWrapper

env = RLWrapper(LuxAIS3GymEnv(numpy_output=True))

check_env(env)

log_path = "logs/ppo_logs"
tensorboard = configure(log_path, ["tensorboard"])
# tensorboard = configure(log_path, ["stdout", "tensorboard"])

model = PPO("MultiInputPolicy", env, verbose=1)
model.set_logger(tensorboard)

model.learn(total_timesteps=1_000_000)
model.save('models/ppo_baseline.bin')
