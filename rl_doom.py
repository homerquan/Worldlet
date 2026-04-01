import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import vizdoom as vzd
except ImportError:
    print("Error: vizdoom is not installed.")
    print("Please install it using: pip install vizdoom")
    sys.exit(1)


class DoomEnv(gym.Env):
    """
    A simple wrapper for ViZDoom to make it compatible with Gymnasium.
    """

    def __init__(self, render=False, scenario="basic"):
        super(DoomEnv, self).__init__()

        self.game = vzd.DoomGame()

        # Look for the default scenarios shipped with the vizdoom package
        scenario_path = os.path.join(
            os.path.dirname(str(vzd.__file__)), "scenarios", f"{scenario}.cfg"
        )

        if not os.path.exists(scenario_path):
            print(f"Error: Could not find scenario file at {scenario_path}")
            sys.exit(1)

        self.game.load_config(scenario_path)
        self.game.set_window_visible(render)

        # Set screen format to RGB and resolution to 160x120
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

        # Enforce standard 5 actions across ALL scenarios
        self.game.clear_available_buttons()
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)

        self.game.init()

        # 5 standard actions: Left, Right, Attack, Forward, Backward
        self.action_space = spaces.Discrete(5)
        self.actions = []
        for i in range(5):
            act = [False] * 5
            act[i] = True
            self.actions.append(act)

        # Observation space is the RGB screen buffer (Height, Width, Channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
        )

    def step(self, action):
        # We skip 4 frames (frame_skip=4)
        reward = self.game.make_action(self.actions[action], 4)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            img = state.screen_buffer
        else:
            img = np.zeros((120, 160, 3), dtype=np.uint8)

        info = {}
        truncated = False
        return img, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        img = state.screen_buffer
        return img, {}

    def close(self):
        self.game.close()
