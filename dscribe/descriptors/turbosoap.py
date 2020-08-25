# -*- coding: utf-8 -*-
import numpy as np
from ase import Atoms
import ase.data
import copy

from dscribe.descriptors import Descriptor
from dscribe.core import System

class TurboSOAP(Descriptor):
    """Class for generating a modified Smooth Overlap of Atomic Orbitals (SOAP)
    many-body atomic descriptor. The implementation differs from the standard SOAP in
    the definition of the underlying atomic density field (the radial and angular
    channels are decoupled in this implementation). In addition to this, higher
    degree of control is available through the introduction of extra hyperparameters.
    TurboSOAP achieves better computational performance and accuracy than the
    standard SOAP. See the references below for further information.

    If you use TurboSOAP, read and cite:
    "Optimizing many-body atomic descriptors for enhanced computational 
    performance of machine learning based interatomic potentials", 
    Miguel Caro, Phys. Rev. B 100, 024112 (2019),
    https://doi.org/10.1103/PhysRevB.100.024112

    In addition, read and cite the original SOAP paper:
    "On representing chemical environments", Albert P. Bartók, Risi Kondor, and
    Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
    https://doi.org/10.1103/PhysRevB.87.184115
    """
    def __init__(self, species, lmax = 8, periodic=False,):
        """
        Each species requires an array of parameters to configure the TurboSOAP descriptor. 
        TurboSOAPSpecie(:class:`turbosoap_dscribe.TurboSOAPSpecie`) provides maximal control
        for each species. Each species *not* defined will use a default value.
        See the TurboSOAPSpecie documentation for detailed parameter definitions.
        Args:
            lmax (int): The maximum degree of spherical harmonics.
            species (dictionary): Dictionary where key is the chemical symbol or atomic number 
                and the value is TurboSOAPSpecie(:class:`turbosoap_dscribe.TurboSOAPSpecie`). If
                the value of key is set to None then the species will have default configuration
            periodic (bool): Determines whether the system is considered to be
                periodic.
        """
        super().__init__(periodic=periodic, flatten=False, sparse=False)
        try:
            from turbosoap_dscribe import prepare_turbosoap_configuration, calculate_turbosoap_descriptor
            self.prepare_turbosoap_configuration = prepare_turbosoap_configuration
            self.calculate_turbosoap_descriptor = calculate_turbosoap_descriptor
        except ImportError:
            raise ImportError("Could not find turbosoap_dscribe module required by TurboSOAP descriptor. "
                "You can install it with 'pip install turbosoap_dscribe'. See documentation for more information")
        atomic_numbers = []
        turbosoap_species = []
        for k,v in species.items():
            if isinstance(k, str):
                if k not in ase.data.atomic_numbers:
                    raise ValueError(
                        f"Provided chemical symbol is not valid: {k}"
                    )
                k = ase.data.atomic_numbers[k]
            else:
                if not (k >= 1 and k < len(ase.data.chemical_symbols)):
                    raise ValueError(
                        f"Provided number is not valid atomic number"
                    )
            atomic_numbers.append(k)
            if v is None:
                v = default_turbosoap_specie
            v_copy = copy.copy(v)
            v_copy.debug_name = ase.data.chemical_symbols[atomic_numbers[-1]]
            turbosoap_species.append(v_copy)
        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError(
                f"Species has multiple definitions: atomic_numbers given: {atomic_numbers}"
            )
        self.atomic_numbers = atomic_numbers
        self.atomic_numbers_to_indices = {k: v for v,k in enumerate(atomic_numbers)}
        self.config = self.prepare_turbosoap_configuration(turbosoap_species, lmax=lmax)
        self.atomic_numbers_to_rcuts = {k: v for k,v in zip(atomic_numbers, self.config['rcut_hard'])}

    def create(self, system, periodic=None):
        """Return the TurboSOAP output for all atoms in the given system.
        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            periodic (bool): Same TurboSOAP can be applied to either periodic or
                non-periodic systems. Use this to override descriptor setting
        Returns:
            np.ndarray: The SOAP output for the given
            systems and positions. The first dimension is determined by the number
            of atoms and systems and the second dimension is determined by
            the get_number_of_features() function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their positions.
        """
        if isinstance(system, (Atoms, System)):
            return self.create_single(system, periodic=periodic)
        systems = system
        num_atoms = [len(s) for s in systems]
        result = np.empty((num_atoms, self.get_number_of_features()), dtype=float)
        atom_idx = 0
        for natoms in num_atoms:
            result[atom_idx:(atom_idx+natoms),:] = self.create_single(system, periodic=periodic)
            atom_idx += natoms
        assert atom_idx == sum(num_atoms)
        return result


    def create_single(self, system, periodic=None):
        """Return the TurboSOAP output for the given system and given positions.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            periodic (bool): Same TurboSOAP can be applied to either periodic or
                non-periodic systems. Use this to override descriptor setting

        Returns:
            np.ndarray  The SOAP output for the
            given system and positions. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features() function.
        """
        if periodic is None:
            periodic = self.periodic
        for atomic_number in system.numbers:
            if atomic_number not in self.atomic_numbers_to_rcuts:
                raise ValueError(f"Atomic number {atomic_number} is found from given system but is not defined in the descriptor")
        return self.calculate_turbosoap_descriptor(self.config, system, periodic,
            self.atomic_numbers_to_indices, self.atomic_numbers_to_rcuts)

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        return self.config['num_components']


class TurboSOAPSpecie:
    """Helper class for all the TurboSOAP per-species parameters
    """
    def __init__(self, rcut, nmax = 8, buffer = 0.5,
        atom_sigma_r = 0.5, atom_sigma_t = 0.5,
        atom_sigma_r_scaling = 0., atom_sigma_t_scaling = 0.,
        radial_enhancement = 0, amplitude_scaling = 1.0,
        central_weight = 1., global_scaling = 1., nf = 4.):
        """
        Args:
            rcut (float):                    A cutoff for local region in angstroms. Should be bigger
                                             than the buffer.

            nmax (int):                      Number of radial basis functions.

            buffer (float):                  Width of buffer region where atomic density field will decay
                                             to zero

            atom_sigma_r (float):            Width of radial sigma (in Angstrom) at the origin

            atom_sigma_r_scaling (float):    Radial-scaling parameter for radial sigma (dimensionless)

            atom_sigma_t (float):            Width of angular sigma (in Angstrom) at the origin

            atom_sigma_t_scaling (float):    Radial-scaling parameter for angular sigma (dimensionless)

            radial_enhancement (integer):    Distant atomic densities get weighted by the integral of
                                             a Gaussian located at the same position times the radial
                                             coordinated raised to this power. It must be 0, 1 or 2

            amplitude_scaling (float):       The atomic densities get weighted, according to the position
                                             of their atomic centers, by a radial function that decays to
                                             zero at the cutoff according to a power law, where this
                                             parameter is the exponent

            global_scaling (float):          Multiplicative factor for the whole atomic density fieldç
                                             corresponding to this species

            central_weight (float):          Scaling factor for the atomic density field corresponding to
                                             the central atom, when the centra latom is of this species

            nf (float):                      Decay parameter of the exponential within the buffer region
        """
        if rcut <= buffer:
            raise ValueError(
                "Rcut should be bigger than the buffer region"
            )
        if buffer <= 0.:
            raise ValueError(
                "The buffer region should be greater than 0"
            )
        self.rcut = rcut
        if nmax < 1 or nmax > 12:
            raise ValueError(
                "nmax should be at least one but at maximum 12"
            )
        self.nmax = nmax
        self.buffer = buffer
        self.atom_sigma_r = atom_sigma_r
        self.atom_sigma_t = atom_sigma_t
        self.atom_sigma_r_scaling = atom_sigma_r_scaling
        self.atom_sigma_t_scaling = atom_sigma_t_scaling
        self.radial_enhancement = radial_enhancement
        self.amplitude_scaling = amplitude_scaling
        self.global_scaling = global_scaling
        self.central_weight = central_weight
        self.nf = nf
        self.debug_name = "default"

default_turbosoap_specie = TurboSOAPSpecie(5.0, 8)
