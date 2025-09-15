import numpy as np


class SimpleField:
    def __init__(self, ground_truth_map, a=1, b=0.015, altitude_range=None, seed=123):
        self.ground_truth_map = ground_truth_map
        self.a = a
        self.b = b
        self.rng = np.random.default_rng(seed)
        if altitude_range is None:
            altitude_range = [10, 60]
        start_alt = altitude_range[0]
        end_alt = altitude_range[-1]
        num_altitudes = 6
        self.altitudes = np.round(np.linspace(start_alt, end_alt, num=num_altitudes), decimals=2)
        self.field_type = "Gaussian"
        self.grid_shape = ground_truth_map.shape

    def get_visible_range(self, uav_pos, fov=60, index_form=False):
        # TODO: make grid_length a parameter
        # TODO: clip out out of boundary
        grid_length = 0.125
        # grid_length = 1 
        # print(f"grid shape: {self.grid_shape}")

        fov_rad = np.deg2rad(fov) / 2

        x_dist = round(uav_pos.altitude * np.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(uav_pos.altitude * np.tan(fov_rad) / grid_length) * grid_length
        # print(f"dist filed x:{x_dist} y:{y_dist}")
        x_min = max(0, uav_pos.position[0] - x_dist)
        x_max = min(self.grid_shape[1], uav_pos.position[0] + x_dist)
        y_min = max(0, uav_pos.position[1] - y_dist)
        y_max = min(self.grid_shape[0], uav_pos.position[1] + y_dist)

        if x_max - x_min == 0 or y_max - y_min == 0:
            return [[0, 0], [0, 0]]

        if not index_form:
            return [[x_min, x_max], [y_min, y_max]]

        # i_min = int(y_min)
        # i_max = int(y_max)
        i_min= int(self.grid_shape[0] - y_max / grid_length)
        i_max= int(self.grid_shape[0] - y_min / grid_length)
        j_min = int(x_min / grid_length)
        j_max = int(x_max / grid_length)
        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, uav_pos, sigmas=None):
        [[i_min, i_max], [j_min, j_max]] = self.get_visible_range(uav_pos, index_form=True)
        # print(f"Footprint indices in filed: {i_min}-{i_max}, {j_min}-{j_max}")
        submap = self.ground_truth_map[i_min:i_max, j_min:j_max]

        if sigmas is None:
            sigma = self.a * (1 - np.exp(-self.b * uav_pos.altitude))
            sigmas = [sigma, sigma]

        sigma0, sigma1 = sigmas[0], sigmas[1]
        random_values = self.rng.random(submap.shape)
        success0 = random_values <= 1.0 - sigma0
        success1 = random_values <= 1.0 - sigma1
        z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
        z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
        observations = np.where(submap == 0, z0, z1)

        fp_vertices_ij = {
            "ul": np.array([i_min, j_min]),
            "bl": np.array([i_max, j_min]),
            "ur": np.array([i_min, j_max]),
            "br": np.array([i_max, j_max]),
        }

        return fp_vertices_ij, observations
