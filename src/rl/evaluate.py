from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode, SingleAgentWrapper
from luxai_s3.params import EnvParams
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from kits.python.agent import Agent
from rl.wrappers import RLWrapper

env = LuxAIS3GymEnv(numpy_output=True)
# env = SingleAgentWrapper(env, 'player_0', Agent)
env = RecordEpisode(env, save_dir="episodes")
env = RLWrapper(env=env)

check_env(env)


# from stable_baselines3 import PPO
# rl_agent = PPO.load('./models/ppo_baseline.bin')

def evaluate_single_agents(seed=42, games_to_play=3, replay_save_dir="logs/replays"):
    env = RLWrapper(
        RecordEpisode(
            LuxAIS3GymEnv(numpy_output=True),
            save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
        )
    )

    obs, info = env.reset(seed=seed)
    for i in range(games_to_play):
        obs, info = env.reset()

        # env_cfg = info["params"]  # only contains observable game parameters
        # agent = Agent("player_0", env_cfg)
        # player_1 = agent_2_cls("player_1", env_cfg)

        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:
            # actions = dict()
            # for agent in [player_0, player_1]:
            #     actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            # actions: {p0:..., p1: ...}

            # random action:
            action = env.action_space.sample()

            # sample agent action
            # action = agent.act(step=step, obs=env.backout_obs(obs))

            ## rl agent action
            # action, _ = rl_agent.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            # info["state"] is the environment state object, you can inspect/play around with it to e.g. print
            # unobservable game data that agents can't see
            game_done = terminated or truncated
            step += 1
    env.close()  # free up resources and save final replay


evaluate_single_agents()
