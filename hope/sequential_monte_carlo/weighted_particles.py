from typing import Optional

import numpy as np
from numpy.typing import NDArray


class ParticleArray:
    """A weighted particle array for sequential Monte Carlo inference.

    Stores particle locations and their associated importance weights, and
    provides resampling methods to refresh the particle set. The
    ``duplicate_ratio`` property is computed lazily and invalidated whenever
    locations change.

    .. note::
        ``duplicate_ratio`` is computed by scanning the full location array for
        unique rows. Computing the duplicate_ratio isn't very efficient in this
        format, since duplicates have to be computed from the multidimensional
        particle array. It would be more efficient to have a separate index array
        that keeps track of which locations are currently represented several
        times. This however would incur bookkeeping with the indeces.
        For now this implementation is probably fast enough.

    Parameters
    ----------
    init_particles : ndarray of shape (n_particles, n_dimensions)
        Initial particle locations. Weights are initialised to
        ``1 / n_particles``.

    Attributes
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimensionality of each particle.
    locations : ndarray of shape (n_particles, n_dimensions)
        Current particle locations.
    weights : ndarray of shape (n_particles,)
        Current importance weights. Always normalized to sum to 1.
    """

    def __init__(self, init_particles):
        self.n_particles = init_particles.shape[0]
        self.n_dim = init_particles.shape[1]
        self.locations = init_particles
        self.weights = np.ones(self.n_particles) / self.n_particles
        self._duplicate_ratio = None

    @property
    def duplicate_ratio(self) -> float:
        """Fraction of particles that are duplicates of another particle.

        Computed lazily and cached until ``locations`` changes. A value of 0
        means all particles are unique; a value approaching 1 means nearly all
        particles are at the same location.

        Returns
        -------
        float
            Proportion of particles in ``[0, 1]`` that are duplicates.
        """
        if self._duplicate_ratio is None:
            unique_locations = np.unique(self.locations, axis=0)
            self._duplicate_ratio = 1 - unique_locations.shape[0] / self.n_particles
        return self._duplicate_ratio

    def update_locations(
        self,
        new_locations: NDArray[np.float64],
        indices_of_updated: NDArray[np.intp],
    ) -> None:
        """Update a subset of particle locations in-place.

        Invalidates the cached ``duplicate_ratio``.

        Parameters
        ----------
        new_locations : ndarray of shape (n_updated, n_dimensions)
            Replacement locations for the updated particles.
        indices_of_updated : ndarray of shape (n_updated,)
            Indices of the particles to update, as returned by (e.g.)
            ``metropolis_step``.
        """

        self._duplicate_ratio = None
        self.locations[indices_of_updated] = new_locations

    def importance_resampling(
        self, method: str = "multinomial", rng: Optional[np.random.Generator] = None
    ) -> None:
        """Resample particles according to their importance weights and reset to uniform weights.

        Draws ``n_particles`` new indices from the current particle set using
        ``weights`` as the sampling distribution, replaces ``locations`` with
        the selected particles, and resets all weights to ``1 / n_particles``.
        Invalidates the cached ``duplicate_ratio``.

        Parameters
        ----------
        method : {"multinomial", "stratified"}
            Resampling scheme.

            ``"multinomial"``
                Each new particle is drawn independently from the categorical
                distribution defined by ``weights``.
            ``"stratified"``
                Divides ``[0, 1]`` into ``n_particles`` equal strata and draws
                one uniform sample per stratum, yielding lower variance than
                multinomial resampling.
        rng : np.random.Generator, optional
            Random number generator. A new default generator is created if
            ``None``.

        Raises
        ------
        NotImplementedError
            If ``method`` is not ``"multinomial"`` or ``"stratified"``.
        """

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