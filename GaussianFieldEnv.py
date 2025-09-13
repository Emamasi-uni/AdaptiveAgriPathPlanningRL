import numpy as np
import gymnasium as gym
from gymnasium import spaces

from filed import SimpleField
from helper import gaussian_random_field, perform_mapping, uav_position
from uav_camera import Camera
from mapper_LBP import OccupancyMap as OM


class GaussianFieldEnv(gym.Env):
    def __init__(self, grid_info, altitudes=6, fov=60, binary=False, cluster_radius=3.0, f_overlap=0.8, s_overlap=0.7, a=1, b=0.015):
        super().__init__()
        self.grid_info = grid_info
        self.altitudes = altitudes
        self.fov = fov
        self.binary_field = binary
        self.cluster_radius = cluster_radius

        ovelap = None
        self.camera = Camera(grid=self.grid_info, fov_angle=fov, f_overlap=ovelap, s_overlap=ovelap, a=a, b=b)

        self.h_step = self.camera.get_hstep()
        self.h_range = self.camera.get_hrange()
        self.xy_step = self.camera.xy_step
        self.x_range = self.camera.x_range
        self.y_range = self.camera.y_range

        self.true_field = None
        self.belief_map = None
        self.observation_count = None
        self.agent_pos = (0.0, 0.0)
        self.agent_altitude = self.h_range[0]

        self.action_space = spaces.Discrete(7)  # 0-6 

        # Placeholder for observation_space (updated in reset)
        self.observation_space = None
        self.initial_entropy = 0.0
        self.reset()

    def reset(self, seed=None):
        self.occupancy_map = OM(self.grid_info.shape)
        self.belief_map = np.full((self.grid_info.shape[0], self.grid_info.shape[1], 2), 0.5)
        ground_truth = gaussian_random_field(self.cluster_radius, self.grid_info.shape, binary=self.binary_field,
                                             seed=seed)
        self.true_field = SimpleField(ground_truth_map=ground_truth, a=1, b=0.015, altitude_range=[10, 60])
        self.observation_count = np.zeros((self.grid_info.shape[0], self.grid_info.shape[1], self.altitudes),
                                          dtype=np.int32)

        # Reset Camera
        self.camera.reset()
        self.agent_pos = self.camera.position
        self.agent_altitude = self.camera.altitude

        # Compute maximum footprint size at minimum altitude
        self.camera.set_altitude(self.h_range[0])  # minimum altitude
        self.fp_shape = self._get_fp_shape()  # store maximum shape
        self.camera.set_altitude(self.agent_altitude)  # restore initial altitude

        pooled_size = self.fp_shape[0] * self.fp_shape[1]
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(pooled_size + pooled_size * self.altitudes + self.altitudes,),
            dtype=np.float32
        )

        self.initial_entropy = self._compute_entropy()
        return self._get_observation(), {}

    def step(self, action_idx):
        actions = ["front", "back", "right", "left", "up", "down", "hover"]
        action = actions[action_idx]

        # Current state
        x = self.camera.get_x()

        # New state
        new_pos, new_alt = self.camera.x_future(action, x)
        self.camera.set_position(new_pos)
        self.camera.set_altitude(new_alt)

        self.agent_pos = new_pos
        self.agent_altitude = new_alt

        # Get observation from camera
        _, observed_submap = self.camera.get_observations(self.true_field.ground_truth_map)
        self._update_observation_count_from_camera()
        uav_pos = uav_position((self.agent_pos, self.agent_altitude))

        self.belief_map, observed_ids, submap, fp_vertices_ij = perform_mapping(
            self.belief_map,
            self.occupancy_map,
            self.true_field,
            self.camera,
            uav_pos,
            conf_dict=None     # pass sensor here if needed
        )

        new_entropy = self._compute_entropy()
        reward = self.initial_entropy - new_entropy
        self.initial_entropy = new_entropy

        return self._get_observation(), reward, False, False, {}

    def _compute_entropy(self):
        # TODO entropy of all cell? or only observed?
        p = self.belief_map
        return -np.sum(p * np.log(p + 1e-10))

    def _update_observation_count_from_camera(self):
        [[i_min, i_max], [j_min, j_max]] = self.camera.get_range(index_form=True)
        altitude_index = int(round((self.agent_altitude - self.h_range[0]) / self.h_step))
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                if 0 <= i < self.grid_info.shape[0] and 0 <= j < self.grid_info.shape[1]:
                    self.observation_count[i, j, altitude_index] += 1

    def _get_fp_shape(self):
        [[i_min, i_max], [j_min, j_max]] = self.camera.get_range(index_form=True)
        return (max(1, i_max - i_min), max(1, j_max - j_min))

    def _get_observation(self):
        [[i_min, i_max], [j_min, j_max]] = self.camera.get_range(index_form=True)
        belief_patch = self.belief_map[i_min:i_max, j_min:j_max]
        count_patch = self.observation_count[i_min:i_max, j_min:j_max, :]

        pooled_belief = self._average_pool(belief_patch, *self.fp_shape)
        pooled_counts = np.stack([
            self._average_pool(count_patch[..., a], *self.fp_shape)
            for a in range(self.altitudes)
        ], axis=-1)

        obs = np.concatenate([
            pooled_belief.flatten(),
            pooled_counts.flatten(),
            self._altitude_one_hot()
        ])
        return obs.astype(np.float32)

    def _altitude_one_hot(self):
        one_hot = np.zeros(self.altitudes)
        idx = int(round((self.agent_altitude - self.h_range[0]) / self.h_step))
        if 0 <= idx < self.altitudes:
            one_hot[idx] = 1.0
        return one_hot

    def _average_pool(self, patch, out_h=None, out_w=None):
        # If patch is 2D, stack with zeros to make it 3D
        if patch.ndim == 2:
            patch = np.stack([patch, np.zeros_like(patch)], axis=-1)
        elif patch.ndim == 3 and patch.shape[2] > 2:
            patch = patch[..., :2]

        # Use maximum stored shape if not provided
        if out_h is None or out_w is None:
            out_h, out_w = self.fp_shape

        in_h, in_w, _ = patch.shape
        pooled = np.zeros((out_h, out_w))
        stride_h = max(1, in_h // out_h)
        stride_w = max(1, in_w // out_w)

        for i in range(out_h):
            for j in range(out_w):
                x_start = min(i * stride_h, in_h)
                y_start = min(j * stride_w, in_w)
                x_end = min(x_start + stride_h, in_h)
                y_end = min(y_start + stride_w, in_w)
                if x_start < in_h and y_start < in_w:
                    pooled[i, j] = np.mean(patch[x_start:x_end, y_start:y_end])
                else:
                    pooled[i, j] = 0.0  # zero padding
        return pooled
