from os import path

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box


class AnymalCEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 33,
    }

    def __init__(self, **kwargs):
        model_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "../assets/anybotics_anymal_c/scene.xml",
        )
        frame_skip = 30  # Control time step is 0.03 s and pyshics time step is 0.001 s
        observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))

        super().__init__(
            model_path=model_xml_file_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs
        )

    def reset_model(self):
        return np.zeros((2), dtype=np.float32)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()
        return np.zeros((2), dtype=np.float32), 0, False, False, {}
