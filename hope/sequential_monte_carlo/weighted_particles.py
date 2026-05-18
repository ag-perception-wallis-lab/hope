import time

import numpy as np

# In this file three different particle classes are implemented and speed tested
# against each other. For now the winner remains the simple ParticleArray. Both
# other classes need more effort to either fix or vectorize (instead of for
# loops).


class ParticleArray:
    """
    A simple class to represent a set of particles and their weights. Computing
    the duplicate_ratio isn't very efficient in this format, since duplicates
    have to be computed from the multidimensional particle array. It would be
    more efficient to have a separate index array that keeps track of which
    locations are currently represented several times. This however would incur
    bookkeeping with the indeces. For now this implementation is probably fast
    enough.
    """

    def __init__(self, init_particles):
        self.n_particles = init_particles.shape[0]
        self.n_dim = init_particles.shape[1]
        self.locations = init_particles
        self.weights = np.ones(self.n_particles) / self.n_particles
        self._duplicate_ratio = None

    @property
    def duplicate_ratio(self):
        # start = time.time()
        if self._duplicate_ratio is None:
            unique_locations = np.unique(self.locations, axis=0)
            self._duplicate_ratio = 1 - unique_locations.shape[0] / self.n_particles
        # end = time.time()
        # print(f"Duplicate ratio took {(end - start) * 1000} milliseconds")
        return self._duplicate_ratio

    def update_locations(self, new_locations, indices_of_updated):
        """
        indices_of_updated : np.ndarray
            The array contains the indices of the particles that were updated.
        """
        # start = time.time()
        self._duplicate_ratio = None  # needs to be recomputed
        self.locations[indices_of_updated] = new_locations
        # end = time.time()
        # print(f"Update locations took {(end - start) * 1000} milliseconds")

    def importance_resampling(
        self, method="multinomial", rng: np.random.Generator = None
    ):
        self._duplicate_ratio = None  # needs to be recomputed
        if not rng:
            rng = np.random.default_rng()
        if method == "multinomial":
            indices = rng.choice(
                self.n_particles, size=self.n_particles, p=self.weights
            )
            self.locations = self.locations[indices]
        elif method == "stratified":
            black_dots = (
                rng.uniform(low=0, high=1, size=self.n_particles)
                + np.arange(self.n_particles)
            ) / self.n_particles
            colored_bars = np.cumsum(self.weights)
            indices = np.digitize(black_dots, bins=colored_bars)
            self.locations = self.locations[indices]
        else:
            raise NotImplementedError
        self.weights = np.ones(self.n_particles) / self.n_particles

    def __str__(self):
        return f"{np.array2string(self.locations)}"


# TODO: Fix broken routine shorten and adapt max_capacity
class ParticleArrayList:
    """
    A class to represent a set of particles and their weights. Instead of having
    a single array that simply contains the locations of the particles (which is
    inefficient when calculating duplicates), this class has an array of
    locations and a separate array of active indices. This way computing
    duplicate_ratio can be done on the indices array.
    However, currently the "shorten" routine is broken.
    """

    def __init__(self, init_particles):
        self.n_particles = init_particles.shape[0]
        self.n_dim = init_particles.shape[1]
        self.capacity = self.n_particles * 2
        self.max_capacity = self.n_particles * 100000
        self._locations = np.zeros((self.capacity, self.n_dim))
        self._locations[: self.n_particles] = init_particles
        self.size = self.n_particles
        self.active_indices = np.arange(self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    @property
    def locations(self):
        return self._locations[self.active_indices]

    @property
    def duplicate_ratio(self):
        # Duplicate ratio of indices is between 0.17 and 0.4 ms for 10000
        # particles.
        # Duplicate ratio of locations is at around 30 ms for 10000
        # particles and 100 dimensions.
        # Conclusion: Duplicate ratio with
        # locations is to slow for our real-time problems and mumbo-jumbo with
        # indices is necessary.

        start = time.time()
        unique_indices = np.unique(self.active_indices)
        ratio = 1 - unique_indices.shape[0] / self.n_particles
        end = time.time()
        print(f"Duplicate ratio with indices took {(end - start) * 1000} milliseconds")
        start = time.time()
        unique_locations = np.unique(self.locations, axis=0)
        loc_ratio = 1 - unique_locations.shape[0] / self.n_particles
        end = time.time()
        print(f"Duplicate ratio with locs took {(end - start) * 1000} milliseconds")
        assert np.isclose(ratio, loc_ratio)
        return ratio

    def update_locations(self, new_locations, indices_of_updated):
        """
        indices_of_updated : np.ndarray
            The array contains the indices of the particles in the old index
            array that were updated. E.g. array([0]), if the first index was
            updated (no matter which number that index was).
        """
        # For 10000 particles, this takes 0.03 and 0.04 ms. When enlarging, it takes 0.3 ms, but order of magnitude grows by 1 every second time!
        start = time.time()
        n_new = new_locations.shape[0]
        if self.size + n_new >= self.capacity:
            print(
                "######################################Enlarging######################################"
            )
            print()
            print()
            self.enlarge()

        self._locations[self.size : self.size + n_new] = new_locations
        self.active_indices[indices_of_updated] = np.arange(n_new) + self.size
        self.size += n_new
        end = time.time()
        print(f"Update locations took {(end - start) * 1000} milliseconds")

    def enlarge(self):
        if self.capacity * 2 > self.max_capacity:
            self.shorten()
        else:
            self.capacity *= 2
            _new_locations_array = np.zeros((self.capacity, self.n_dim))
            _new_locations_array[: self.size] = self._locations[: self.size]
            self._locations = _new_locations_array

    def shorten(self):
        print(
            "\n\n\n################################\nShorten!!!!!!!!!\n#################################\n\n\n\n"
        )
        self.capacity = self.n_particles * 2
        _new_locations_array = np.zeros((self.capacity, self.n_dim))
        _new_locations_array[: self.n_particles] = self.locations
        self.size = self.n_particles
        self.active_indices = np.arange(self.n_particles)

    def importance_resampling(self, method="multinomial", seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        if method == "multinomial":
            self.active_indices = np.random.choice(
                self.active_indices, size=self.n_particles, p=self.weights
            )
        elif method == "stratified":
            black_dots = (
                np.random.uniform(low=0, high=1, size=self.n_particles)
                + np.arange(self.n_particles)
            ) / self.n_particles
            colored_bars = np.cumsum(self.weights)
            indices_for_index_array = np.digitize(black_dots, bins=colored_bars)
            print(indices_for_index_array)
            self.active_indices = self.active_indices[indices_for_index_array]
        else:
            raise NotImplementedError
        self.weights = np.ones(self.n_particles) / self.n_particles

    def __str__(self):
        return f"{np.array2string(self.locations)}"  # ,,,,, \n {np.array2string(self._locations)} \n"


class InPlaceParticleArrayList:
    """
    This class similarly to ParticleArrayList represents particles with two
    arrays. One with the locations and one with the corresponding active
    indices. However instead of ParticleArrayList's expanding locations array,
    it has a fixed size locations array that is updated in place. Since updating
    locations is done in a for loop for now, updating locations is too slow for
    now.
    """

    def __init__(self, init_particles):
        self.n_particles = init_particles.shape[0]
        self.n_dim = init_particles.shape[1]
        self._locations = init_particles
        self.active_indices = np.arange(self.n_particles)
        self.free_indices = []
        self.weights = np.ones(self.n_particles) / self.n_particles

    @property
    def locations(self):
        return self._locations[self.active_indices]

    @property
    def duplicate_ratio(self):
        unique_indices = np.unique(self.active_indices)
        return 1 - unique_indices.shape[0] / self.n_particles

    def update_locations(self, new_locations, indices_of_updated):
        """
        indices_of_updated : np.ndarray
            The array contains the indices of the particles in the old index
            array that were updated. E.g. array([0]), if the first index was
            updated (no matter which number that index was).
        """

        start = time.time()
        n_new = new_locations.shape[0]
        unique_old_indices, inverse, counts = np.unique(
            self.active_indices, return_inverse=True, return_counts=True
        )
        for new_loc, ind in zip(new_locations, indices_of_updated):
            if counts[inverse[ind]] == 1:
                self._locations[ind] = new_loc
                counts[inverse[ind]] -= 1
            elif counts[inverse[ind]] > 1:
                next_ind = self.free_indices.pop()
                self._locations[next_ind] = new_loc
                self.active_indices[ind] = next_ind
                counts[inverse[ind]] -= 1
            elif counts[inverse[ind]] == 0:
                raise ValueError("This should not happen")
        end = time.time()
        print(f"Update locations took {(end - start) * 1000} milliseconds")

    def importance_resampling(self, method="multinomial", seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        if method == "multinomial":
            self.active_indices = np.random.choice(
                self.active_indices, size=self.n_particles, p=self.weights
            )
        elif method == "stratified":
            black_dots = (
                np.random.uniform(low=0, high=1, size=self.n_particles)
                + np.arange(self.n_particles)
            ) / self.n_particles
            colored_bars = np.cumsum(self.weights)
            indices_for_index_array = np.digitize(black_dots, bins=colored_bars)
            print(indices_for_index_array)
            self.active_indices = self.active_indices[indices_for_index_array]
        else:
            raise NotImplementedError
        self.free_indices = list(
            np.setdiff1d(
                np.arange(self.n_particles),
                self.active_indices,
            )
        )
        self.weights = np.ones(self.n_particles) / self.n_particles

    def __str__(self):
        return f"{np.array2string(self.locations)}"  # ,,,,, \n {np.array2string(self._locations)} \n"
