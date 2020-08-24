
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.operators import op_converter
import numpy,time


def h2(dist=0.75):
    mol = PySCFDriver(atom=
                      'H 0.0 0.0 0.0;'\
                      'H 0.0 0.0 {}'.format(dist), unit=UnitsType.ANGSTROM, charge=0,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    h1 = mol.one_body_integrals
    h2 = mol.two_body_integrals
    
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy
    num_particles = mol.num_alpha + mol.num_beta + 0
    ferOp = FermionicOperator(h1=h1, h2=h2)
    qubitOp = ferOp.mapping(map_type='parity',threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
    
    
    cHam = op_converter.to_matrix_operator(qubitOp)
    cHam = cHam.dense_matrix + nuclear_repulsion_energy*numpy.identity(4)

    return cHam

def hehp(dist=1.0):
    mol = PySCFDriver(atom=
                      'He 0.0 0.0 0.0;'\
                      'H 0.0 0.0 {}'.format(dist), unit=UnitsType.ANGSTROM, charge=1,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    h1 = mol.one_body_integrals
    h2 = mol.two_body_integrals
    
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy
    num_particles = mol.num_alpha + mol.num_beta + 0
    ferOp = FermionicOperator(h1=h1, h2=h2)
    qubitOp = ferOp.mapping(map_type='parity',threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
    
    
    cHam = op_converter.to_matrix_operator(qubitOp)
    cHam = cHam.dense_matrix + nuclear_repulsion_energy*numpy.identity(4)

    return cHam

def lih(dist=1.5):
    mol = PySCFDriver(atom=
                      'H 0.0 0.0 0.0;'\
                      'Li 0.0 0.0 {}'.format(dist), unit=UnitsType.ANGSTROM, charge=0,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    freeze_list = [0]
    remove_list = [-3, -2]
    repulsion_energy = mol.nuclear_repulsion_energy
    num_particles = mol.num_alpha + mol.num_beta
    num_spin_orbitals = mol.num_orbitals * 2
    remove_list = [x % mol.num_orbitals for x in remove_list]
    freeze_list = [x % mol.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + mol.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + mol.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=mol.one_body_integrals, h2=mol.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    
    qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    
    shift = energy_shift + repulsion_energy
    cHam = op_converter.to_matrix_operator(qubitOp)
    cHam = cHam.dense_matrix + shift*numpy.identity(16)

    return cHam
