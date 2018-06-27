from __future__ import absolute_import, division, print_function
from builtins import super
import math
import numpy as np
import itertools

from scipy.spatial.distance import squareform, pdist, cdist
from scipy.sparse import lil_matrix, coo_matrix
from scipy.special import erf

from describe.core import System
from describe.descriptors import Descriptor


class Orbital(Descriptor):
    """Implementation of the Many-body tensor representation up to K=3.
    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems, please always use a primitive cell. It does not
    matter which of the available primitive cell is used.
    """
    decay_factor = math.sqrt(2)*3

    def __init__(
            self,
            k,
            periodic,
            atomic_numbers,
            grid=None,
            weighting=None,
            flatten=True
            ):
        """
        Args:
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Notice that this is
                not the atomic numbers that are present for an individual
                system, but should contain all the elements that are ever going
                to be encountered when creating the descriptors for a set of
                systems.  Keeping the number of handled elements as low as
                possible is preferable.
            k (set or list): The interaction terms to consider from 1 to 3. The
                size of the final output and the time taken in creating this
                descriptor is exponentially dependent on this value.
            periodic (bool): Boolean for if the system is periodic or none. If
                this is set to true, you should provide the primitive system as
                input and then the number of periodic copies is determined from the
                'cutoff'-values specified in the weighting argument.
            grid (dictionary): This dictionary can be used to precisely control
                the broadening width, grid spacing and grid length for all the
                different terms. If not provided, a set of sensible defaults
                will be used. Example:
                    grid = {
                        "k1": {
                            "min": 1,
                            "max": 10
                            "sigma": 0.1
                            "n": 100
                        },
                        "k2": {
                            "min": 0,
                            "max": 1/0.70,
                            "sigma": 0.01,
                            "n": 100
                        },
                        ...
                    }

                Here 'min' is the minimum value of the axis, 'max' is the
                maximum value of the axis, 'sigma' is the standard devation of
                the gaussian broadening and 'n' is the number of points sampled
                on the grid.
            weighting (dictionary or string): A dictionary of weighting functions and an
                optional threshold for each term. If None, weighting is not
                used. Weighting functions should be monotonically decreasing.
                The threshold is used to determine the minimum mount of
                periodic images to consider. If no explicit threshold is given,
                a reasonable default will be used.  The K1 term is
                0-dimensional, so weighting is not used. You can also use a
                string to indicate a certain preset. The available presets are:

                    'exponential':
                        weighting = {
                            "k2": {
                                "function": lambda x: np.exp(-0.5*x),
                                "threshold": 1e-3
                            },
                            "k3": {
                                "function": lambda x: np.exp(-0.5*x),
                                "threshold": 1e-3
                            }
                        }

                The meaning of x changes for different terms as follows:
                    K=1: x = 0
                    K=2: x = Distance between A->B
                    K=3: x = Distance from A->B->C->A.
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array. If False, a list of the different tensors is
                provided.

        Raises:
            ValueError if the given k value is not supported, or the weighting
            is not specified for periodic systems.
        """
        super().__init__(flatten)

        # Check K value
        supported_k = set([2])
        if isinstance(k, int):
            raise ValueError(
                "Please provide the k values that you wish to be generated as a"
                " list or set."
            )
        else:
            try:
                k = set(k)
            except Exception:
                raise ValueError(
                    "Could not make the given value of k into a set. Please "
                    "provide the k values as a list or a set."
                )
            if not k.issubset(supported_k):
                raise ValueError(
                    "The given k parameter '{}' has at least one invalid k value".format(k)
                )
            self.k = set(k)

        # Check the weighting information
        if weighting is not None:
            if weighting == "exponential":
                weighting = {
                    "k2": {
                        "function": lambda x: np.exp(-0.5*x),
                        "threshold": 1e-3
                    },
                    "k3": {
                        "function": lambda x: np.exp(-0.5*x),
                        "threshold": 1e-3
                    }
                }
            else:
                for i in self.k:
                    info = weighting.get("k{}".format(i))
                    if info is not None:
                        assert "function" in info, \
                            ("The weighting dictionary is missing 'function'.")
        elif periodic:
            raise ValueError(
                "Periodic systems will need to have a weighting function "
                "defined in the 'weighting' dictionary of the MBTR constructor."
            )

        # Check the given grid
        if grid is not None:
            for i in self.k:
                info = grid.get("k{}".format(i))
                if info is not None:
                    msg = "The grid information is missing the value for {}"
                    val_names = ["min", "max", "sigma", "n"]
                    for val_name in val_names:
                        try:
                            info[val_name]
                        except Exception:
                            raise KeyError(msg.format(val_name))

                    # Make the n into integer
                    n = grid.get("k{}".format(i))["n"]
                    grid.get("k{}".format(i))["n"] = int(n)
                    assert info["min"] < info["max"], \
                        "The min value should be smaller than the max values"
        self.grid = grid

        self.present_elements = None
        self.atomic_number_to_index = {}
        self.atomic_number_to_d1 = {}
        self.atomic_number_to_d2 = {}
        self.index_to_atomic_number = {}
        self.n_atoms_in_cell = None
        self.periodic = periodic
        self.n_copies_per_axis = None
        self.weighting = weighting

        # Sort the atomic numbers. This is not needed but makes things maybe a
        # bit easier to debug.
        atomic_numbers.sort()
        for i_atom, atomic_number in enumerate(atomic_numbers):
            self.atomic_number_to_index[atomic_number] = i_atom
            self.index_to_atomic_number[i_atom] = atomic_number
        self.n_elements = len(atomic_numbers)

        self.max_atomic_number = max(atomic_numbers)
        self.min_atomic_number = min(atomic_numbers)

        # These are the orbitals that are present
        self.orbitals = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p"]

        self._counts = None
        self._inverse_distances = None
        self._angles = None
        self._angle_weights = None
        self._axis_k1 = None
        self._axis_k2 = None
        self._axis_k3 = None

    def describe(self, system):
        """Return the many-body tensor representation as a 1D array for the
        given system.

        Args:
            system (System): The system for which the descriptor is created.

        Returns:
            1D ndarray: The many-body tensor representation up to the k:th term
            as a flattened array.
        """
        self.n_atoms_in_cell = len(system)
        present_element_numbers = set(system.numbers)
        self.present_indices = set()
        for number in present_element_numbers:
            index = self.atomic_number_to_index[number]
            self.present_indices.add(index)

        mbtr = []
        if 2 in self.k:

            settings_k2 = self.get_k2_settings()

            # If needed, create the extended system
            system_k2 = system
            if self.periodic:
                system_k2 = self.create_extended_system(system, 2)

            k2 = self.K2(system_k2, settings_k2)

            # Free memory
            system_k2 = None

            mbtr.append(k2)

        if self.flatten:
            length = 0

            datas = []
            rows = []
            cols = []
            for tensor in mbtr:
                size = tensor.shape[1]
                coo = tensor.tocoo()
                datas.append(coo.data)
                rows.append(coo.row)
                cols.append(coo.col + length)
                length += size

            datas = np.concatenate(datas)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            final_vector = coo_matrix((datas, (rows, cols)), shape=[1, length], dtype=np.float32)

            return final_vector
        else:
            return mbtr

    def get_k2_settings(self):
        """Returns the min, max, dx and sigma for K2.
        """
        if self.grid is not None and self.grid.get("k2") is not None:
            return self.grid["k2"]
        else:
            sigma = 2**(-7)
            min_k = 0-Orbital.decay_factor*sigma
            max_k = 1/0.7+Orbital.decay_factor*sigma
            return {
                "min": min_k,
                "max": max_k,
                "sigma": sigma,
                "n": int(math.ceil((max_k-min_k)/sigma/4) + 1),
            }

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        n_orbitals = len(self.orbitals)

        if 2 in self.k:
            n_k2_grid = self.get_k2_settings()["n"]
            n_k2 = (n_orbitals*(n_orbitals+1)/2)*n_k2_grid
            n_features += n_k2

        return int(n_features)

    def create_extended_system(self, primitive_system, term_number):
        """Used to create a periodically extended system, that is as small as
        possible by rejecting atoms for which the given weighting will be below
        the given threshold.

        Args:
            primitive_system (System): The original primitive system to
                duplicate.
            term_number (int): The term number of the tensor. For k=2, the max
                distance is x, for k>2, the distance is given by 2*x.

        Returns:
            System: The new system that is extended so that each atom can at
            most have a weight that is larger or equivalent to the given
            threshold.
        """
        numbers = primitive_system.numbers
        relative_pos = primitive_system.get_scaled_positions()
        cartesian_pos = np.array(primitive_system.get_positions())
        cell = primitive_system.get_cell()

        # Determine the upper limit of how many copies we need in each cell
        # vector direction. We take as many copies as needed for the
        # exponential weight to come down to the given threshold.
        cell_vector_lengths = np.linalg.norm(cell, axis=1)
        n_copies_axis = np.zeros(3, dtype=int)
        weighting_function = self.weighting["k{}".format(term_number)]["function"]
        threshold = self.weighting["k{}".format(term_number)].get("threshold", 1e-3)

        for i_axis, axis_length in enumerate(cell_vector_lengths):
            limit_found = False
            n_copies = -1
            while (not limit_found):
                n_copies += 1
                distance = n_copies*cell_vector_lengths[0]

                # For terms above k==2 we double the distances to take into
                # account the "loop" that is required.
                if term_number > 2:
                    distance = 2*distance

                weight = weighting_function(distance)
                if weight < threshold:
                    n_copies_axis[i_axis] = n_copies
                    limit_found = True

        # Create copies of the cell but keep track of the atoms in the
        # original cell
        num_extended = []
        pos_extended = []
        num_extended.append(numbers)
        pos_extended.append(cartesian_pos)
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        for i in range(-n_copies_axis[0], n_copies_axis[0]+1):
            for j in range(-n_copies_axis[1], n_copies_axis[1]+1):
                for k in range(-n_copies_axis[2], n_copies_axis[2]+1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    num_copy = np.array(numbers)

                    # Calculate the positions of the copied atoms and filter
                    # out the atoms that are farther away than the given
                    # cutoff.
                    pos_copy = np.array(relative_pos)-i*a-j*b-k*c
                    pos_copy_cartesian = np.dot(pos_copy, cell)
                    distances = cdist(pos_copy_cartesian, cartesian_pos)

                    # For terms above k==2 we double the distances to take into
                    # account the "loop" that is required.
                    if term_number > 2:
                        distances *= 2

                    weights = weighting_function(distances)
                    weight_mask = weights >= threshold

                    # Create a boolean mask that says if the atom is within the
                    # range from at least one atom in the original cell
                    valids_mask = np.any(weight_mask, axis=1)

                    valid_pos = pos_copy_cartesian[valids_mask]
                    valid_num = num_copy[valids_mask]

                    pos_extended.append(valid_pos)
                    num_extended.append(valid_num)

        pos_extended = np.concatenate(pos_extended)
        num_extended = np.concatenate(num_extended)

        extended_system = System(
            positions=pos_extended,
            numbers=num_extended,
            cell=cell,
        )

        return extended_system

    def gaussian_sum(self, centers, weights, settings):
        """Calculates a discrete version of a sum of Gaussian distributions.

        The calculation is done through the cumulative distribution function
        that is better at keeping the integral of the probability function
        constant with coarser grids.

        The values are normalized by dividing with the maximum value of a
        gaussian with the given standard deviation.

        Args:
            centers (1D np.ndarray): The means of the gaussians.
            weights (1D np.ndarray): The weights for the gaussians.
            settings (dict): The grid settings

        Returns:
            Value of the gaussian sums on the given grid.
        """
        start = settings["min"]
        stop = settings["max"]
        sigma = settings["sigma"]
        n = settings["n"]

        max_val = 1/(sigma*math.sqrt(2*math.pi))

        dx = (stop - start)/(n-1)
        x = np.linspace(start-dx/2, stop+dx/2, n+1)
        pos = x[np.newaxis, :] - centers[:, np.newaxis]
        y = weights[:, np.newaxis]*1/2*(1 + erf(pos/(sigma*np.sqrt(2))))
        f = np.sum(y, axis=0)
        f /= max_val
        f_rolled = np.roll(f, -1)
        pdf = (f_rolled - f)[0:-1]/dx  # PDF is the derivative of CDF

        return pdf

    def elements(self, system):
        """Calculate the atom count for each element.

        Args:
            system (System): The atomic system.

        Returns:
            1D ndarray: The counts for each element in a list where the index
            of atomic number x is self.atomic_number_to_index[x]
        """
        numbers = system.numbers
        unique, counts = np.unique(numbers, return_counts=True)
        counts_reindexed = np.zeros(self.n_elements)
        for atomic_number, count in zip(unique, counts):
            index = self.atomic_number_to_index[atomic_number]
            counts_reindexed[index] = count

        self._counts = counts_reindexed
        return counts_reindexed

    def get_axis(self, k):
        """Return the axis used for the kth term.
        """
        if k == 2:
            settings = self.get_k2_settings()
            minimum = settings["min"]
            maximum = settings["max"]
            n = settings["n"]
            axis = np.linspace(minimum, maximum, n)
        return axis

    def inverse_distances(self, system):
        """Calculates the inverse distances for the given atomic positions.

        Args:
            system (System): The atomic system.

        Returns:
            dict: Inverse distances in the form:
            {i: { j: [list of angles] }}. The dictionaries are filled
            so that the entry for pair i and j is in the entry where j>=i.
        """
        inverse_dist = system.get_inverse_distance_matrix()
        numbers = system.numbers
        inv_dist_dict = {}

        for i_atom, i_element in enumerate(numbers):
            for j_atom, j_element in enumerate(numbers):
                if j_atom > i_atom:
                    # Only consider pairs that have one atom in the original
                    # cell
                    if i_atom < self.n_atoms_in_cell or \
                       j_atom < self.n_atoms_in_cell:

                        i_index = self.atomic_number_to_index[i_element]
                        j_index = self.atomic_number_to_index[j_element]

                        # Make sure that j_index >= i_index so that we fill only
                        # the upper triangular part
                        i_index, j_index = sorted([i_index, j_index])

                        old_dict = inv_dist_dict.get(i_index)
                        if old_dict is None:
                            old_dict = {}
                        old_list = old_dict.get(j_index)
                        if old_list is None:
                            old_list = []
                        inv_dist = inverse_dist[i_atom, j_atom]
                        old_list.append(inv_dist)
                        old_dict[j_index] = old_list
                        inv_dist_dict[i_index] = old_dict

        self._inverse_distances = inv_dist_dict
        return inv_dist_dict

    def K2(self, system, settings):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Args:
            system (System): The atomic system.
            settings (dict): The grid settings

        Returns:
            1D ndarray: flattened K2 values.
        """
        start = settings["min"]
        stop = settings["max"]
        n = settings["n"]
        self._axis_k2 = np.linspace(start, stop, n)

        inv_dist_dict = self.inverse_distances(system)
        n_orbitals = len(self.orbitals)
        n_elem = self.n_elements

        # if self.flatten:
            # k2 = lil_matrix(
                # (1, int(n_orbitals*(n_orbitals+1)/2*n)), dtype=np.float32)
        # else:
        k2 = np.zeros((n_orbitals, n_orbitals, n))

        # Determine the weighting function
        weighting_function = None
        if self.weighting is not None and self.weighting.get("k2") is not None:
            weighting_function = self.weighting["k2"]["function"]

        # import json
        # with open("../data/orbital_map.json", "r") as fin:
            # orbital_map = json.load(fin)
        import describe.data.orbital_map
        orbital_map = describe.data.orbital_map.orbital_map

        for i in range(n_elem):
            for j in range(n_elem):
                if j >= i:
                    try:
                        inv_dist = np.array(inv_dist_dict[i][j])
                    except KeyError:
                        continue

                    # Calculate weights from distance
                    if weighting_function is not None:
                        dist_weights = weighting_function(1/np.array(inv_dist))
                    else:
                        dist_weights = np.ones(len(inv_dist))

                    # See which elements are in question
                    i_elem = self.index_to_atomic_number[i]
                    j_elem = self.index_to_atomic_number[j]

                    # Get the orbitals associated with each element
                    i_orbitals = orbital_map[i_elem]
                    j_orbitals = orbital_map[j_elem]

                    # For each pair of orbitals, add an entry into the orbital
                    # map weighted by the weighting function and sum of
                    # electrons in these orbitals
                    n_i_orbitals = len(i_orbitals)
                    n_j_orbitals = len(j_orbitals)
                    for i_orbital, i_n_electrons in enumerate(i_orbitals):
                        for j_orbital, j_n_electrons in enumerate(j_orbitals):

                            if i_orbital == n_i_orbitals - 1 and j_orbital == n_j_orbitals - 1:

                                # Get the weight coming from number of electrons
                                electron_weight = i_n_electrons + j_n_electrons

                                # Broaden with a gaussian
                                gaussian_sum = self.gaussian_sum(inv_dist, dist_weights*electron_weight, settings)

                                # if self.flatten:
                                    # i_start = i_orbital*n_orbitals - np.sum(np.arange(0, i_orbital)) + j_orbital
                                    # start = i_start*n
                                    # end = (i_start + 1)*n
                                    # k2[0, start:end] += gaussian_sum
                                # else:
                                k2[i_orbital, j_orbital, :] += gaussian_sum

        k2_flat = lil_matrix(
            (1, int(n_orbitals*(n_orbitals+1)/2*n)), dtype=np.float32
        )
        m = -1
        if self.flatten:
            for i in range(n_orbitals):
                for j in range(n_orbitals):
                    if j >= i:
                        m += 1
                        start = m*n
                        end = (m + 1)*n
                        k2_flat[0, start:end] = k2[i, j, :]
            k2 = k2_flat

        return k2
