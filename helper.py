import numpy as np


def perform_mapping(
        belief_map,
        occupancy_map,
        field_map,
        camera,
        uav_pos,
        conf_dict=None
):
    """
    Updates the belief map based on current observations.

    Parameters
    ----------
    belief_map : np.ndarray
        Belief map (H, W, 2) with [p(0), p(1)].
    occupancy_map : OM
        Occupancy Map instance used to update beliefs.
    field_map : Field-like
        Object with method get_observations(uav_pos, sigmas).
    camera : Camera
        Camera object to compute the observed footprint.
    uav_pos : uav_position
        Current UAV state (position, altitude).
    conf_dict : dict, optional
        Sensor parameters to determine standard deviations (sigma).

    Returns
    -------
    belief_map : np.ndarray
        Updated belief map.
    observed_ids : set
        Cells observed in this step.
    submap : np.ndarray
        Local observed map.
    fp_vertices_ij : list
        Footprint vertices in grid coordinates.
    """

    sigmas = None
    if conf_dict is not None:
        s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
        sigmas = [s0, s1]

    fp_vertices_ij, submap = field_map.get_observations(uav_pos, sigmas)

    occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)

    belief_map[:, :, 1] = occupancy_map.get_belief().copy()
    belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

    observed_ids = observed_m_ids(camera, uav_pos)

    return belief_map, observed_ids, submap, fp_vertices_ij


def gaussian_random_field(cluster_radius, n_cell, binary=False, seed=123):
    n_cell_x, n_cell_y = n_cell

    def _fft_indices(n):
        a = list(range(0, int(np.floor(n / 2)) + 1))
        b = reversed(range(1, int(np.floor(n / 2))))
        b = [-i for i in b]
        return a + b

    def _pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        val = np.sqrt(np.sqrt(kx ** 2 + ky ** 2) ** (-cluster_radius))
        return val

    map_rng = np.random.default_rng(seed)
    amplitude = np.zeros((n_cell_x, n_cell_y))
    fft_indices_x = _fft_indices(n_cell_x)
    fft_indices_y = _fft_indices(n_cell_y)

    for i, kx in enumerate(fft_indices_x):
        for j, ky in enumerate(fft_indices_y):
            amplitude[i, j] = _pk2(kx, ky)

    noise = np.fft.fft2(map_rng.normal(size=(n_cell_x, n_cell_y)))
    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (
            np.max(random_field) - np.min(random_field)
    )

    if binary:
        normalized_random_field[normalized_random_field >= 0.5] = 1
        normalized_random_field[normalized_random_field < 0.5] = 0
        normalized_random_field = normalized_random_field.astype(np.uint8)

    return normalized_random_field


def observed_m_ids(uav=None, uav_pos=None, aslist=True):
    if uav != None and uav_pos != None:
        [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = uav.get_range(
            position=uav_pos.position, altitude=uav_pos.altitude, index_form=True
        )
    else:
        raise TypeError("Pass either z or uav_position")
    if aslist:

        observed_m = []
        for i_b in range(obsd_m_i_min, obsd_m_i_max):
            for j_b in range(obsd_m_j_min, obsd_m_j_max):
                observed_m.append((i_b, j_b))
        return observed_m
    else:
        return [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]]


def collect_sample_set(grid):
    # Create an array of central cells for each 3x3 block (using slices)
    rows, cols = grid.shape

    valid_rows = (rows // 3) * 3
    valid_cols = (cols // 3) * 3

    # remove remainer rows and cols % 3
    truncated_grid = grid[:valid_rows, :valid_cols]

    central_cells = truncated_grid[1::3, 1::3]

    # Create a matrix of neighbors for each central cell using slicing
    north = truncated_grid[0::3, 1::3]  # One row above central cells
    south = truncated_grid[2::3, 1::3]  # One row below central cells
    west = truncated_grid[1::3, 0::3]  # One column to the left
    east = truncated_grid[1::3, 2::3]  # One column to the right

    neighbors = np.stack([north, south, west, east], axis=-1)

    neighbor_sums = np.sum(neighbors, axis=-1)

    return np.column_stack((central_cells.flatten(), neighbor_sums.flatten()))


def pearson_correlation_coeff(d_sampled):
    c_values = d_sampled[:, 0]  # Central cell values
    n_values = d_sampled[:, 1]  # Neighbor sums

    avg_c = np.mean(c_values)
    avg_n = np.mean(n_values)

    # Vectorized calculations for the Pearson correlation
    c_diff = c_values - avg_c
    n_diff = n_values - avg_n

    numerator = np.sum(c_diff * n_diff)
    sum_sq_central_diff = np.sum(c_diff ** 2)
    sum_sq_neighbors_diff = np.sum(n_diff ** 2)

    denominator = np.sqrt(sum_sq_central_diff * sum_sq_neighbors_diff)

    return numerator / denominator if denominator != 0 else 0


def adaptive_weights_matrix(obs_map):
    d_sampled = collect_sample_set(obs_map)
    p = pearson_correlation_coeff(d_sampled)
    exp = np.exp(-p)
    psi = np.array(
        [
            [1 / (1 + exp), exp / (1 + exp)],  # For (m_i=0, m_j=0) and (m_i=0, m_j=1)
            [exp / (1 + exp), 1 / (1 + exp)],  # For (m_i=1, m_j=0) and (m_i=1, m_j=1)
        ]
    )
    return psi


class uav_position:
    def __init__(self, input) -> None:
        self.position = input[0]
        self.altitude = input[1]

    def __eq__(self, other):
        if isinstance(other, uav_position):
            return self.position == other.position and self.altitude == other.altitude
        return False

    def __hash__(self):
        return hash((self.position, self.altitude))
