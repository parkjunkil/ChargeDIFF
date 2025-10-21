import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools
from functools import lru_cache
import time

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.core import Element
from networkx.algorithms.components import is_connected
from torch.utils.data import Dataset, DataLoader
#from sklearn.metrics import accuracy_score, recall_score, precision_score

from torch_geometric.data import Data, Batch

from torch_scatter import scatter
from torch_scatter import segment_coo, segment_csr

from p_tqdm import p_umap

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pyxtal.symmetry import Group
from pyxtal import pyxtal

from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool
from tqdm import tqdm 
from functools import partial 

from collections import Counter

import re

import faulthandler
faulthandler.enable()

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from pymatgen.core.composition import Composition

import smact
from smact.screening import pauling_test

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

Hydrogen = ['H']
Alkali_Metal = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
Alkali_Earth_Metal = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
Metalloids = ['B', 'Si', 'Ge', 'As', 'Sb', 'Te']
Reactive_Nonmetals = ['C', 'N', 'Cl', 'Se', 'Br', 'I']
TM_3rd_period = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
TM_4th_period = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
TM_5th_period = ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
TM_6th_period = ['Rf', 'Db', 'Sg', 'Bh', 'Hs']
Post_Transition_Metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']
Lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
Actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
Noble_gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
Oxide = ['O']
Sulfide = ['S']
Flouride = ['F']
Phosphate = ['P']


EPSILON = 1e-5

# Distribution of the number of atoms for each dataset
train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    
    'mp_20' : [0.0,
                0.002357389599477173,
                0.022570254878162638,
                0.021566613761553544,
                0.16130613388105686,
                0.04978526748202782,
                0.0880169918775091,
                0.023037064699841285,
                0.08222855008869387,
                0.032723368499673236,
                0.10001400429465036,
                0.012790589113994959,
                0.09151806554009896,
                0.02238353094949118,
                0.06565680141910185,
                0.015801512463822238,
                0.06290262347119784,
                0.009056110540565774,
                0.04882830734758659,
                0.010970030809448231,
                0.0764867892820465],

    'mp_20_charge' : [0.0,
                0.0026409319774903742,
                0.022040675288774805,
                0.0216210879652483,
                0.14697897127060913,
                0.049042353638068914,
                0.08579326685753776,
                0.023768387797413366,
                0.08164675683680521,
                0.03235758712607365,
                0.10173758515154507,
                0.013081251851120544,
                0.09354329153914503,
                0.02302793957942541,
                0.06829400730575574,
                0.016191134366669958,
                0.06654161318985093,
                0.009502418797512093,
                0.0513871063283641,
                0.011328857735215718,
                0.07947477539737388],

    'mp_20_charge_TMO' : 
                    [   0,
                        0,
                        0.013725490196078431,
                        0.0058823529411764705,
                        0.03137254901960784,
                        0,
                        0.043137254901960784,
                        0.00784313725490196,
                        0.047058823529411764,
                        0.03333333333333333,
                        0.10196078431372549,
                        0.01568627450980392,
                        0.15098039215686274,
                        0.041176470588235294,
                        0.25098039215686274,
                        0.0196078431372549,
                        0.049019607843137254,
                        0.013725490196078431,
                        0.08823529411764706,
                        0.021568627450980392,
                        0.06470588235294118,
                        ],

    'FM_3M' :  [0,
                0.008271735304470257,
                0.0376627947905667,
                0.014959521295318549,
                0.5851812741992256,
                0,
                0,
                0.00017599436818021823,
                0.006863780359028511,
                0.024287222808870117,
                0.30746216121084124,
                0,
                0,
                0.0024639211545230554,
                0.0015839493136219642,
                0.004223864836325237,
                0.0028159098908834917,
                0.0017599436818021823,
                0.0015839493136219642,
                0,
                0.0007039774727208729],
    
    'C2_M' : [0,
                0.00048007681228996637,
                0.003840614498319731,
                0.009121459433509362,
                0.02256361017762842,
                0.007201152184349496,
                0.09601536245799328,
                0.07105136821891503,
                0.03696591454632741,
                0.041766682669227076,
                0.10513682189150264,
                0.044167066730676906,
                0.11809889582333173,
                0.05472875660105617,
                0.10177628420547287,
                0.05472875660105617,
                0.09073451752280365,
                0.014882381180988958,
                0.04704752760441671,
                0.02784445511281805,
                0.05184829572731637],

    'I4_MMM' : [0,
                0.005952380952380952,
                0.0027056277056277055,
                0.05952380952380952,
                0.19264069264069264,
                0.3658008658008658,
                0.057900432900432904,
                0.0817099567099567,
                0.021645021645021644,
                0.022727272727272728,
                0.03463203463203463,
                0.01406926406926407,
                0.04220779220779221,
                0.047619047619047616,
                0.008658008658008658,
                0.007034632034632035,
                0.00487012987012987,
                0.016233766233766232,
                0.00487012987012987,
                0.003787878787878788,
                0.005411255411255411],

    'Pnma' : [0,
                0,
                0,
                0,
                0.0005892751915144372,
                0,
                0,
                0,
                0.06187389510901591,
                0,
                0,
                0,
                0.4779021803182086,
                0,
                0,
                0,
                0.14967589864466707,
                0,
                0,
                0,
                0.309958750736594],

    'R_3m' : [0,
                0.004839685420447671,
                0.030852994555353903,
                0.04597701149425287,
                0.27828191167574107,
                0.03690260133091349,
                0.08650937689050212,
                0.049606775559588624,
                0.06533575317604355,
                0.0012099213551119178,
                0.019358741681790685,
                0.006049606775559589,
                0.0514216575922565,
                0.05263157894736842,
                0.0998185117967332,
                0.03690260133091349,
                0.044162129461584994,
                0.0012099213551119178,
                0.019358741681790685,
                0.038112522686025406,
                0.03145795523290986],

    'P6_3_mmc' : [0,
                0,
                0.029696969696969697,
                0,
                0.06484848484848485,
                0,
                0.18424242424242424,
                0,
                0.45636363636363636,
                0,
                0.08303030303030302,
                0,
                0.1484848484848485,
                0,
                0.013939393939393939,
                0,
                0.012121212121212121,
                0,
                0.0018181818181818182,
                0,
                0.005454545454545455],

    'Cmcm' : [0,
                0,
                0.007407407407407408,
                0,
                0.0962962962962963,
                0,
                0.1111111111111111,
                0,
                0.23185185185185186,
                0,
                0.12296296296296297,
                0,
                0.16666666666666666,
                0,
                0.08222222222222222,
                0,
                0.12074074074074075,
                0,
                0.03037037037037037,
                0,
                0.03037037037037037],

    'Pm_3m' : [0,
                0.0024193548387096775,
                0.225,
                0,
                0.3951612903225806,
                0.3282258064516129,
                0.0016129032258064516,
                0.01935483870967742,
                0.012096774193548387,
                0,
                0,
                0,
                0.0024193548387096775,
                0.0024193548387096775,
                0,
                0.0064516129032258064,
                0.004838709677419355,
                0,
                0,
                0,
                0],

    'P_1' : [0,
                0,
                0,
                0,
                0.0019801980198019802,
                0,
                0.0039603960396039604,
                0.00891089108910891,
                0.03564356435643564,
                0.01089108910891089,
                0.04356435643564356,
                0.01287128712871287,
                0.08415841584158416,
                0.02574257425742574,
                0.08712871287128712,
                0.060396039603960394,
                0.14455445544554454,
                0.05643564356435644,
                0.17524752475247524,
                0.0504950495049505,
                0.19801980198019803],
}

Max_Atom_Num = 100

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')


@lru_cache
def get_atomic_number(symbol: str) -> int:
    # get atomic number from Element symbol
    return Element(symbol).Z


@lru_cache
def get_element_symbol(Z: int) -> str:
    # get Element symbol from atomic number
    return str(Element.from_Z(Z=Z))


def extract_unique_elements(formula):
    element_pattern = re.compile(r'[A-Z][a-z]?')  # 원소 기호 정규식
    elements = element_pattern.findall(formula)
    return elements

# For atom type classfication
classes = [Hydrogen, Alkali_Metal, Alkali_Earth_Metal, Metalloids, Reactive_Nonmetals, TM_3rd_period, TM_4th_period, TM_5th_period, TM_6th_period, Post_Transition_Metals, Lanthanides, Actinides, Noble_gases, Oxide, Sulfide, Flouride, Phosphate]


def atom_classification(chemical_system):
    
    chemical_system_condition = torch.zeros(len(classes)) 

    elements = extract_unique_elements(chemical_system)

    for i in range(len(classes)):
        if any(elem in elements for elem in classes[i]):
            chemical_system_condition[i] = 1.0

    return chemical_system_condition


from pymatgen.analysis.local_env import CrystalNN as crystalnn
def contains_phosphate(structure, cutoff=4):
    cnn = crystalnn()
    phosphate_found = False

    for i, site in enumerate(structure):
        if site.specie.symbol == 'P':
            try:
                neighbors = cnn.get_nn_info(structure, i)
                o_neighbors = [n['site'] for n in neighbors if n['site'].specie.symbol == 'O']
                if len(o_neighbors) == cutoff:
                    phosphate_found = True
                    break
            except:
                continue

    return phosphate_found

            
def multihot_embed(formula):
    elements = extract_unique_elements(formula)
    
    chemical_system_numbers: torch.LongTensor = torch.tensor(
    [get_atomic_number(symbol=_element) for _element in elements], dtype=int)
    
    chemical_system_condition = torch.zeros(Max_Atom_Num)
    
    chemical_system_condition[chemical_system_numbers] = 1.0
    
    return chemical_system_condition
            

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal

def refine_spacegroup(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.
    sym_info = {
        'anchors':anchors,
        'wyckoff_ops':matrices,
        'spacegroup':space_group
    }
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        try:
            #crystal_graph = StructureGraph.with_local_env_strategy(
            crystal_graph = StructureGraph.from_local_env_strategy(
                crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10)
            #crystal_graph = StructureGraph.with_local_env_strategy(
            crystal_Graph = StructureGraph.from_local_env_strategy(
                crystal, crystalNN_tmp) 
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res



def preprocess(input_file, num_workers, niggli, primitive, graph_method,
               property_list, use_space_group = False, tol=0.01):
    df = pd.read_csv(input_file)

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [property_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                       for idx in range(len(df))]

    return ordered_results

def process_one(row, niggli, primitive, graph_method, prop_list, use_space_group = False, tol=0.01):
        
    crystal_str = row['cif']
    
    crystal = build_crystal(
        crystal_str, niggli=niggli, primitive=primitive)
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol = tol)
        result_dict.update(sym_info)
    else:
        result_dict['spacegroup'] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
        
    result_dict.update({
        'mp_id': row['material_id'],
        'cif': crystal_str,
        'graph_arrays': graph_arrays
    })
    result_dict.update(properties)
    return result_dict



def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array['frac_coords']
        atom_types = crystal_array['atom_types']
        lengths = crystal_array['lengths']
        angles = crystal_array['angles']
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
        }
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)
            
        dict['scaled_lattice'] = np.concatenate([lengths, angles])

def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def reorder_symmetric_edges(edge_index, neighbors, keep_self_loops=True):
    num_structures = neighbors.size(0)

    # Identify edges to keep
    i, j = edge_index
    mask_diff = i < j  # i != j, keep one direction only

    if keep_self_loops:
        mask_same = (i == j)
        mask = mask_diff | mask_same
    else:
        mask = mask_diff

    edge_index_new = edge_index[:, mask]

    # Remove duplicate pairs: keep only (i < j), then add (j, i)
    src, dst = edge_index_new[0], edge_index_new[1]
    edge_index_pairs = torch.stack([
        torch.stack([src, dst]),
        torch.stack([dst, src])
    ], dim=2)  # shape: (2, N, 2)

    # Interleave
    edge_index_sym = edge_index_pairs.permute(2, 1, 0).reshape(2, -1)

    # Recalculate neighbors per structure
    old_batch_edge = torch.repeat_interleave(
        torch.arange(num_structures, device=edge_index.device),
        neighbors,
    )

    # Make sure mask is applied to old_batch_edge
    batch_edge = old_batch_edge[mask]  # This will have the same length as mask
    neighbors_new = 2 * torch.bincount(batch_edge, minlength=num_structures)

    return edge_index_sym, neighbors_new


def graph_between_node_probe(pos_atoms, num_atoms_per_batch, voxel_resolution):

    device = pos_atoms.device
    num_probes = voxel_resolution ** 3

    # Step 1: PBC wrap
    atom_frac = pos_atoms % 1.0  # (N, 3)

    # Step 2: Per-atom batch index
    batch_atom_idx = torch.arange(len(num_atoms_per_batch), device=device).repeat_interleave(num_atoms_per_batch)  # (N,)

    # Step 3: Voxel index of each atom (0~7)
    voxel_idx = torch.floor(atom_frac * voxel_resolution).long()  # (N, 3)

    # Step 4: Get 27-neighbor voxel offsets
    offsets = torch.tensor([[i, j, k] for i in [-1, 0, 1]
                                      for j in [-1, 0, 1]
                                      for k in [-1, 0, 1]], device=device)  # (27, 3)

    # (N, 1, 3) + (1, 27, 3) → (N, 27, 3)
    neighbor_voxel = (voxel_idx[:, None, :] + offsets[None, :, :]) % voxel_resolution  # (N, 27, 3)

    # (N, 27) → flat probe indices
    probe_ids = (neighbor_voxel[..., 0] * voxel_resolution**2 +
                 neighbor_voxel[..., 1] * voxel_resolution +
                 neighbor_voxel[..., 2])  # (N, 27)

    # Add batch offset: probe_ids + batch_idx * num_probes
    probe_ids += batch_atom_idx[:, None] * num_probes  # (N, 27)

    # Build src/dst
    N = pos_atoms.shape[0]
    src = torch.arange(N, device=device).repeat_interleave(27)  # (N*27,)
    dst = probe_ids.reshape(-1)  # (N*27,)

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def graph_between_node_probe_large(pos_atoms, num_atoms_per_batch, voxel_resolution):

    device = pos_atoms.device
    num_probes = voxel_resolution ** 3

    # Step 1: PBC wrap
    atom_frac = pos_atoms % 1.0  # (N, 3)

    # Step 2: Per-atom batch index
    batch_atom_idx = torch.arange(len(num_atoms_per_batch), device=device).repeat_interleave(num_atoms_per_batch)  # (N,)

    # Step 3: Voxel index of each atom (0~7)
    voxel_idx = torch.floor(atom_frac * voxel_resolution).long()  # (N, 3)

    # Step 4: Get 125-neighbor voxel offsets
    offsets = torch.tensor([[i, j, k] for i in [-2, -1, 0, 1, 2]
                                      for j in [-2, -1, 0, 1, 2]
                                      for k in [-2, -1, 0, 1, 2]], device=device)  # (125, 3)

    # (N, 1, 3) + (1, 125, 3) → (N, 125, 3)
    neighbor_voxel = (voxel_idx[:, None, :] + offsets[None, :, :]) % voxel_resolution  # (N, 125, 3)

    # (N, 125) → flat probe indices
    probe_ids = (neighbor_voxel[..., 0] * voxel_resolution**2 +
                 neighbor_voxel[..., 1] * voxel_resolution +
                 neighbor_voxel[..., 2])  # (N, 125)

    # Add batch offset: probe_ids + batch_idx * num_probes
    probe_ids += batch_atom_idx[:, None] * num_probes  # (N, 125)

    # Build src/dst
    N = pos_atoms.shape[0]
    src = torch.arange(N, device=device).repeat_interleave(125)  # (N*125,)
    dst = probe_ids.reshape(-1)  # (N*125,)

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    Filters out edges so that each atom has at most `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors per atom
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)

    # Build offsets
    max_num_neighbors = num_neighbors.max()
    index_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_offset_expand = torch.repeat_interleave(index_offset, num_neighbors)

    # Map each edge to a flattened matrix of [atom, neighbor_idx]
    local_index = torch.arange(len(index), device=device) - index_offset_expand
    flat_index = index * max_num_neighbors + local_index

    # Build padded distance matrix
    distance_sort = torch.full((num_atoms * max_num_neighbors,), np.inf, device=device)
    distance_sort[flat_index] = atom_distance
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort distances
    sorted_distance, sorted_idx = torch.sort(distance_sort, dim=1)
    
    topk_mask = (
        torch.arange(max_num_neighbors, device=device)
        .unsqueeze(0)
        .expand(num_atoms, -1) < max_num_neighbors_threshold
    )
    # Flatten mask and indices
    sorted_idx_flat = sorted_idx + index_offset.view(-1,1)
    keep_index = sorted_idx_flat[topk_mask].flatten()
    

    mask = torch.zeros_like(index, dtype=torch.bool)
    
    mask[keep_index] = True

    # Count number of neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    new_num_neighbors = segment_csr(topk_mask.sum(dim=1), image_indptr)

    return mask, new_num_neighbors


def radius_graph_pbc(pos, lengths, angles, natoms, radius, max_num_neighbors_threshold, device, lattices=None):
    
    # device = pos.device
    batch_size = len(natoms)

    cell = lattices
    # position of the atoms
    atom_pos = pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1,1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1,1)
    
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1,1)
    
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1) # N_graphs * 3
#     reps = torch.cat([rep_a1.reshape(-1,1), rep_a2.reshape(-1,1), rep_a3.reshape(-1,1)], dim=1) # N_graphs * 3
    
    unit_cell_all = []
    num_cells_all = []

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    
    unit_cell = torch.cat([_.reshape(-1,1) for _ in torch.meshgrid(cells_per_dim)], dim=-1)
    
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

#     # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    
    
    radius_real = (min_dist.min(dim=-1)[0] + 0.01)#.clamp(max=radius)
    
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    # print(min_dist.min(dim=-1)[0])
    
    # radius_real = radius
    
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
        
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)

    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(
            natoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))
    #unique_edge_index = torch.unique(edge_index, dim=1)

    return edge_index, unit_cell, num_neighbors_image
    #return unique_edge_index



def radius_graph_between_sets_pbc(pos_set1, pos_set2, lengths, angles, num_set1_atoms, num_set2_atoms,
                                radius, max_num_neighbors_threshold, device, lattices=None):
    """
    PBC-aware radius graph between two different sets of atoms (e.g., atoms and charges).

    Args:
        pos_set1 (Tensor): (N1, 3) fractional coordinates of set 1 (e.g., atoms)
        pos_set2 (Tensor): (N2, 3) fractional coordinates of set 2 (e.g., charges)
        lengths, angles: lattice parameters
        num_set1_atoms (Tensor): number of atoms in set 1 for each structure (e.g., [n1, n2, ...])
        num_set2_atoms (Tensor): number of atoms in set 2 (e.g., [512, 512, ...])
        radius (float): cutoff radius
        max_num_neighbors_threshold (int): max neighbors per atom
        device (torch.device): torch device
        lattices (Tensor): (B, 3, 3) lattice matrices

    Returns:
        edge_index: LongTensor of shape (2, E)
        unit_cell: Tensor of shape (E, 3)
    """
    batch_size = len(num_set1_atoms)
    if lattices is None:
        raise ValueError("Lattices must be provided directly")
    cell = lattices

    index_offset_set1 = torch.cumsum(num_set1_atoms, dim=0) - num_set1_atoms
    index_offset_set2 = torch.cumsum(num_set2_atoms, dim=0) - num_set2_atoms

    pair_counts = num_set1_atoms * num_set2_atoms
    total_pairs = pair_counts.sum()

    index_sqr_offset = torch.cumsum(pair_counts, dim=0) - pair_counts
    atom_pair_idx = torch.arange(total_pairs, device=device)

    expand_factor1 = torch.repeat_interleave(num_set1_atoms, pair_counts)
    expand_factor2 = torch.repeat_interleave(num_set2_atoms, pair_counts)
    
    batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), pair_counts)

    index1 = (atom_pair_idx // expand_factor2) #+ index_offset_set1[batch_idx]
    index2 = (atom_pair_idx % expand_factor2) + index_offset_set2[batch_idx]

    pos1 = pos_set1[index1]
    pos2 = pos_set2[index2]

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    min_dist_a1 = (1 / torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)).reshape(-1, 1)
    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    min_dist_a2 = (1 / torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)).reshape(-1, 1)
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    min_dist_a3 = (1 / torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)).reshape(-1, 1)

    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1)
    max_rep = torch.ones(3, dtype=torch.long, device=device)


    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep]
    unit_cell = torch.cat([_.reshape(-1, 1) for _ in torch.meshgrid(cells_per_dim, indexing="ij")], dim=-1)
    num_cells = len(unit_cell)

    unit_cell_per_pair = unit_cell.view(1, num_cells, 3).repeat(pos1.shape[0], 1, 1)
    cell_batch = cell[batch_idx]

    pbc_offsets = torch.bmm(cell_batch, unit_cell.T.unsqueeze(0).repeat(cell_batch.size(0), 1, 1))
    
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    
    pos2 = pos2 + pbc_offsets

    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
        
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    
    
    radius_real = (min_dist.min(dim=-1)[0] + 0.01)#.clamp(max=radius)
    
    radius_real = torch.repeat_interleave(radius_real, pair_counts * num_cells)

    # print(min_dist.min(dim=-1)[0])
    
    # radius_real = radius
    
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_pair.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=num_set1_atoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
            
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=num_set1_atoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(
            num_set1_atoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(num_set1_atoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))
    #unique_edge_index = torch.unique(edge_index, dim=1)

    return edge_index, unit_cell, num_neighbors_image
    #return unique_edge_index



class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
    
    
def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


class Crystal(object):

    def __init__(self, crys_array_dict, cutoff=0.5):
        
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        self.cutoff = cutoff
        
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    
    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure, self.cutoff)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    
    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)
    
    
def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff):
    
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

    
def get_crystal_array_list(data, batch_idx=0):

    if isinstance(data, str):
        data = torch.load(data, map_location='cpu')
  

    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
            
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 


def get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    
    
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list

def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 


class SampleDataset(Dataset):

    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = np.array(train_dist[dataset])
                
        if self.distribution.sum() != 1:
            self.distribution = self.distribution / self.distribution.sum() # For special cases such as phosphate generation
                
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon_24'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data



class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )