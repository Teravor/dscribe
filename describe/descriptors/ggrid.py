import numpy as np

import ase.build
from ase import Atoms
from ase.visualize import view
import ase.data

import matid.geometry

from describe.descriptors.descriptor import Descriptor


class GGrid(Descriptor):
    def __init__(self, a, n, channels, threshold, flatten=False, periodic=True):
        """
        Args:
            a (float): Lattice constant for the cubic volume.
            n (integer): Number of grid points per dimension.
            offset (np.ndarray): A 3D vector specifying an offset between the
                origin of the original system and the origin of the resulting grid.
            channels (list of dict): A list containing dictionaries that
                specify an amplitude and standard deviation for each element.
                Each dictionary in this list represents a new channel
            threshold (float): Determines how remote Gaussians affect the grid
                results. When the Gaussian has decayed below this value, it is
                excluded from the grid values.
            flatten (bool): Whether to flatten the output.
            periodic (bool): Whether the system is periodically repeated within
                the box.
        """
        super().__init__(flatten)
        self.a = a
        self.n = n
        self.channels = channels
        self.threshold = threshold
        self.periodic = periodic

    def get_number_of_features(self):
        """Returns the number of features in this descriptor. Each channel has
        (n x n x n) inputs.
        """
        n_channels = len(self.channels)
        return n_channels*self.n**3

    def get_shape(self):
        """Returns dimensions of the unflattened output.
        """
        n_channels = len(self.channels)
        return [n_channels, self.n, self.n, self.n]

    def create(self, system, offset=None):
        if offset is None:
            offset = np.array((0, 0, 0))
        self.offset = offset
        return super().create(system)

    def describe(self, atoms, offset=None):
        """Creates a cubic volume that has been filled with repetitions of the
        given periodic system, where atoms have been replaced by gaussians of
        width and height determined by the channel information.

        Args:
            atoms(System): The atomic system that is used to fill the
            volume.

        Returns:
            np.ndarray: 3D grid
        """
        if offset is not None:
            self.offset = offset

        # Determine the padding based on the convergence of the gaussians.
        max_pad = 0
        for channel in self.channels:
            for element, params in channel.items():

                # Here we solve the distance at which the gaussian has decayed
                # to the threshold value
                amplitude = params["amplitude"]
                std = params["std"]

                # If the term under square root is negative, or the amplitude
                # is zero, the gaussian is always below the threshold
                if amplitude != 0:
                    sqrt_term = -2*np.log((std*np.sqrt(2*np.pi)*self.threshold)/amplitude)
                    if sqrt_term >= 0:
                        i_pad = std*np.sqrt(sqrt_term)
                        if i_pad > max_pad:
                            max_pad = i_pad
        # print(max_pad)

        # Define the wanted cell that also has approppriate padding
        cell = np.array([
            [self.a + 2*max_pad, 0, 0],
            [0, self.a + 2*max_pad, 0],
            [0, 0, self.a + 2*max_pad],
        ])

        # Find all atom locations that will affect the insides of the cell
        # throught their gaussian "cloud"
        locations = []
        indices = []
        searched_indices = set()
        pad_offset = np.array([max_pad, max_pad, max_pad])
        self.find(atoms, pad_offset+self.offset, np.array([0, 0, 0]), cell, locations, indices, searched_indices, True)
        if self.periodic:
            while len(indices) != 0:
                index = indices.pop()
                self.find(atoms, pad_offset+self.offset, index, cell, locations, indices, searched_indices, False)

        # Gather the atomic positions and atomic numbersl
        atom_pos = np.array([x[0] for x in locations])
        atom_pos = np.vstack(atom_pos)
        atomic_numbers = np.array([x[1] for x in locations])
        atomic_numbers = np.hstack(atomic_numbers)

        # Debug by viewing with ASE
        # atoms = Atoms(
            # cell=cell,
            # symbols=atomic_numbers,
            # positions=atom_pos,
            # pbc=False,
        # )
        # view(atoms)

        # Here we create a 4D array of locations on a grid
        amin = 0
        amax = self.a
        nj = complex(0, self.n)
        x, y, z = np.mgrid[amin:amax:nj, amin:amax:nj, amin:amax:nj]
        grid = np.zeros((self.n, self.n, self.n, 3), dtype=float)
        grid[:, :, :, 0] = x
        grid[:, :, :, 1] = y
        grid[:, :, :, 2] = z

        # The amplitudes are defined by the channel values
        grids = []

        # The positions are shifted from the origin of the padded grid to the
        # origin of the final grid
        shifted_atom_pos = atom_pos-pad_offset

        # The 3D grid is created for each channel
        for channel in self.channels:

            amplitudes = np.array([channel[ase.data.chemical_symbols[x]]["amplitude"] for x in atomic_numbers])
            stds = np.array([channel[ase.data.chemical_symbols[x]]["std"] for x in atomic_numbers])

            # Evaluate the sum of all gaussians at the center of every voxel
            sums = self.gaussian_sum(grid, shifted_atom_pos, stds, amplitudes)

            grids.append(sums)

        grids = np.array(grids)
        return grids

    def find(self, system, offset, index, volume, locations, indices, searched_indices, first_cell):
        tuple_index = tuple(index)
        if tuple_index not in searched_indices:
            origin = np.dot(index, system.get_cell())
            # print(origin)
            pos = origin + system.get_positions() + offset
            pos_cell_basis = matid.geometry.change_basis(pos, volume)
            mask = (pos_cell_basis >= 0) & (pos_cell_basis <= 1)
            inside_mask = np.all(mask, axis=1)
            inside_indices = np.where(inside_mask)[0]
            new_locs_found = len(inside_indices) != 0

            # Add the found positions to a shared list
            if new_locs_found:
                symbols = system.get_atomic_numbers()[inside_indices]
                positions = pos[inside_indices]
                locations.append((positions, symbols))

            # If the parent was valid or locations are found, continue to
            # neighbours
            if first_cell or new_locs_found:
                multipliers = matid.geometry.cartesian([[0, 1, -1], [0, 1, -1], [0, 1, -1]])[1:, :]
                multipliers += index
                for ind in multipliers:
                    indices.append(ind)

        searched_indices.add(tuple_index)

    def gaussian_sum(self, grid, centers, stds, amplitudes):
        """
        """
        # Calculate distance from center on each grid location
        displacements = grid[None, :] - centers[:, None, None, None, :]

        # Reduce xyz to distance only
        distances = np.linalg.norm(displacements, axis=-1)

        # Evaluate gaussian on each point. The Gaussian is normalized so that it
        # reaches 1 at the maximum.

        # Broadcast standard deviations and amplitude to fit the grid
        stds = stds[:, None, None, None]
        amplitudes = amplitudes[:, None, None, None]

        gaussians = 1/(stds*np.sqrt(2*np.pi))*amplitudes*np.exp(-distances**2 / (2 * stds**2))

        # Sum gaussians
        sums = np.sum(gaussians, axis=0)

        return sums
