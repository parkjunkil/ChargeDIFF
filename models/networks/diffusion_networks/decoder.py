# Reference: A large portion of the model architecture is adapted from the repos: https://github.com/jiaor17/DiffCSP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.data_util import graph_between_node_probe

MAX_ATOMIC_NUM=100

# Fourier transformation to process distances between nodes
class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MPLayer(nn.Module):
    """ Message passing layer """

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        chgden_resol=8,
        ln=False,
        ip=True
    ):
        super(MPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        
        self.hidden_dim = hidden_dim
        self.chgden_resol = chgden_resol
        
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        
        ### MLP for A<->A message passing
        self.edge_AA_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 12 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)

        ### MLP for A->P message passing
        self.edge_AP_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 12 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
            
         ### MLP for P->A message passing
        self.edge_PA_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 12 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        
        ### MLP within atom model
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        
        ### MLP within probe model
        self.probe_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,)

        ### Replacing message passing between Probe nodes (P<->P)
        self.probe_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 4, kernel_size=1),
            act_fn,
            nn.Conv3d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1, padding_mode='circular'),
            act_fn,
            nn.Conv3d(hidden_dim // 4, hidden_dim, kernel_size=1),
            
        )
        ### Layer normalzation
        self.ln = ln
        if self.ln:
            self.layer_norm_atom = nn.LayerNorm(hidden_dim)
            self.layer_norm_charge = nn.LayerNorm(hidden_dim)
    
    
    ### compute message between source and target nodes
    def compute_edge_features(self, src, tgt, edge_index, frac_diff, lattices, edge2graph, edge_mlp):
        
        hi = src[edge_index[0]]
        hj = tgt[edge_index[1]]

        # sinusoidal embedding for distance 
        if self.dis_emb is not None:
            frac_diff_embed = self.dis_emb(frac_diff)
            
        # inner product of lattice parameters to keep invariance (reference: https://github.com/jiaor17/DiffCSP)
        lattice_ips = lattices @ lattices.transpose(-1, -2) if self.ip else lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        
        # cosine similarity between fractional coordinate and lattice has been incorporated following https://github.com/microsoft/mattergen
        reshaped_frac_diff = frac_diff.unsqueeze(1)
        cosine = torch.cosine_similarity(reshaped_frac_diff, lattices[edge2graph], dim=-1)

        edge_input = torch.cat([hi, hj, frac_diff_embed, lattice_ips_flatten_edges, cosine], dim=1)
        edge_feats = edge_mlp(edge_input)
        
        return edge_feats

    # Update atom node features after message passing ('atom node' is doenoted simply as 'node')
    def update_nodes(self, node_features, edge_features_AA, edge_features_PA, edge_index_AA, edge_index_PA, mlp):    
         
        agg_AA = scatter(edge_features_AA, edge_index_AA[1], dim = 0, reduce='mean', dim_size=node_features.shape[0]) # (# atom, 512)
        agg_PA = scatter(edge_features_PA, edge_index_PA[1], dim = 0, reduce='mean', dim_size=node_features.shape[0]) # (# atom, 512)          
        agg = torch.cat([node_features, agg_AA, agg_PA], dim = 1)
        
        return node_features + mlp(agg)
    
    # Update probe node features after message passing  ('probe node' is doenoted simply as 'probe')  
    def update_probes(self, probe_features, edge_features_AP, edge_index_AP, mlp):

        agg_AP = scatter(edge_features_AP, edge_index_AP[1], dim = 0, reduce='mean', dim_size=probe_features.shape[0]) # (# probe, 512)
        agg = torch.cat([probe_features, agg_AP], dim = 1)
        probe_features = probe_features + mlp(agg)
        
        # Probe-Probe interaction via CNN    
        probe_features_out = probe_features.view(-1, self.chgden_resol, self.chgden_resol, self.chgden_resol, self.hidden_dim).permute(0, 4, 1, 2, 3) # voxelize
        probe_features_out = self.probe_conv(probe_features_out)
        probe_features_out = probe_features_out.permute(0, 2, 3, 4, 1).reshape(-1, self.hidden_dim) # flatten back
        
        return probe_features_out
    

    def forward(self, node_features, probe_features, lattices, graph_dict):
        
        # layer norm  
        if self.ln:
            node_features = self.layer_norm_atom(node_features)
            probe_features = self.layer_norm_charge(probe_features)
               
        # prepare messages for message passing                                     
        edge_features_AA = self.compute_edge_features(node_features, node_features, graph_dict['edge_index_AA'], graph_dict['frac_diff_AA'], lattices, graph_dict['edge_graph_AA'], self.edge_AA_mlp)
        edge_features_AP = self.compute_edge_features(node_features, probe_features, graph_dict['edge_index_AP'], graph_dict['frac_diff_AP'], lattices, graph_dict['edge_graph_AP'], self.edge_AP_mlp)
        edge_features_PA = self.compute_edge_features(probe_features, node_features, graph_dict['edge_index_PA'], graph_dict['frac_diff_PA'], lattices, graph_dict['edge_graph_PA'], self.edge_PA_mlp)
    
        # update node features as a result of message passing  
        updated_node_features = self.update_nodes(node_features, edge_features_AA, edge_features_PA, graph_dict['edge_index_AA'], graph_dict['edge_index_PA'], self.node_mlp)
        updated_probe_features = self.update_probes(probe_features, edge_features_AP, graph_dict['edge_index_AP'], self.probe_mlp)
        
        return updated_node_features, updated_probe_features

# equivariant grapth neural network for denoisng network
class CHARGEDIFFNet(nn.Module):

    def __init__(
        self,
        chgden_resol = 8,          # latent charge density resolution (8x8x8)
        chgden_dim = 3,            # dimension for latent charge density (3)
        hidden_dim = 128,          # hyperparameter for layer dimension
        latent_dim = 256,          # hyperparameter for layer dimension
        num_layers = 4,            # number of message passing layers
        max_atoms = 100,           # total number of atom types used for one-hot encodding
        act_fn = 'silu',           # activation function used within message passing
        dis_emb = 'sin',           # distance embedding between nodes (Fourier Transformation)
        num_freqs = 10,            # hyperparameter for distance embedding
        edge_style_atom = 'fc',    # message-passing type definition (fully-connected)
        edge_style_charge = 'knn', # message-passing type definition (k-neareset neighborhood)
        expansion_dim = 40,        # hyperparameter for CNN layers for charge density
        ln = False,                # to enable layer normalization
        ip = True,                 # inner product for lattice parameter (set it always True to satisfy invariance)
        pred_type = True,          # set to True for de novo generation 
        smooth = False,            # regarding node embedding within the decoder (set to False)
    ):
        super(CHARGEDIFFNet, self).__init__()

        
        # node embedding to process one-hot encoded atom type vector
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        self.chgden_resol = chgden_resol # resolution for latent voxel (8x8x8)
        self.chgden_dim = chgden_dim # dimension for vqvae latent voxel (3)
        self.num_layers = num_layers # number of message passing layers
        self.coord_out = nn.Linear(hidden_dim*2, 3, bias = False) # readout layer for atomic coordinate
        self.charge_out = nn.Linear(hidden_dim, self.chgden_dim, bias = False) # readout layer for charge density
        self.lattice_out = nn.Linear(hidden_dim*2, 9, bias = False) # readout layer for lattice parameter
        self.ln = ln # enable layer norm
        self.ip = ip # lattice parameter inner product (True)
        self.edge_style_atom = edge_style_atom # set message passing type (fully connected vs kNN)
        self.edge_style_charge = edge_style_charge # set message passing type (fully connected vs kNN)
        self.expansion_dim = expansion_dim # hyperparametr for latent vector dimension
        self.pred_type = pred_type # whether to consider atom type or not (True)
        
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        
         # formulate distance embedding between nodes
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        
        # formulate message passing layer
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, MPLayer(hidden_dim, self.act_fn, self.dis_emb, self.chgden_resol, ln=ln, ip=ip)
            )            
        
        # define layers for layer normalization 
        if self.ln:
            self.final_layer_norm_A = nn.LayerNorm(hidden_dim)
            self.final_layer_norm_P = nn.LayerNorm(hidden_dim)
            
        # if False, it becomes crystal structure prediction tasks (currently not covered)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim*2, MAX_ATOMIC_NUM)
        
        # set up coordinates for latent charge density voxel (8x8x8)
        probe_indices = torch.arange(chgden_resol)
      
        x = probe_indices / chgden_resol + 1/(chgden_resol * 2) # (8,)
        y = probe_indices / chgden_resol + 1/(chgden_resol * 2) # (8,)
        z = probe_indices / chgden_resol + 1/(chgden_resol * 2) # (8,)

        xx, yy, zz = torch.meshgrid(x, y, z)
        frac_coords_probe = torch.stack([xx, yy, zz], dim=-1)
        self.frac_coords_probe = frac_coords_probe.view(-1, 3)

        # convolution layers for probe<->probe interaction
        self.charge_tot_latent_emb =  nn.Sequential(
            nn.Conv3d(3, expansion_dim // 4, kernel_size=3, padding=1, padding_mode='circular'),
            self.act_fn,
            nn.Conv3d(expansion_dim // 4, expansion_dim, kernel_size=1),
        )

        # layer to process probe nodes
        self.probe_latent_emb = nn.Linear(expansion_dim + latent_dim, hidden_dim)

    # prepare edge features between atom nodes
    def gen_edges_AA(self, num_atoms, frac_coords):

        # fully-connected
        if self.edge_style_atom == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            rel_vec = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1. 
            rel_vec = rel_vec - (rel_vec > 0.5).float() # relative fractional distance with considering PBCs
            return fc_edges, rel_vec
        
        else:
            # For atom-atom graph formation, only the fully-connected approach is considered 
            # This is because 1) the number of atoms is managable (i.e., less than 20)
            #                 2) if atom-atom interactions are ignored, there  is a risk of losing crtically important information
            # However, to apply the model to systems with a larger number of atoms, the KNN algorithm can be easily implmented
            raise Exception("Unexpected graph type")
    
    # prepare edge features between atom nodes and probe nodes (A->P & P->A) 
    def gen_edges_PA(self, num_atoms, num_probes, frac_coords_atoms, frac_coords_probes):

        # fully-connected (not recommended!)
        if self.edge_style_charge == 'fc':
            # Caution: Fully connected approach can be significantly slower than KNN
            # It is recommended to use the KNN algorithm instead 

            device = frac_coords_atoms.device
            B = num_atoms.shape[0]

            atom_offset = torch.cumsum(F.pad(num_atoms[:-1], (1, 0)), dim=0)  # [B]
            probe_offset = torch.cumsum(F.pad(num_probes[:-1], (1, 0)), dim=0)  # [B]

            src_list = []
            dst_list = []

            for b in range(B):
                n_atoms = num_atoms[b]
                n_probes = num_probes[b]
                
                a_offset = atom_offset[b]
                p_offset = probe_offset[b]
                
                probe_ids = torch.arange(n_probes, device=device) + p_offset  # [n_probes]
                atom_ids = torch.arange(n_atoms, device=device) + a_offset    # [n_atoms]

                src = probe_ids.repeat_interleave(n_atoms)  # [n_probes * n_atoms]
                dst = atom_ids.repeat(n_probes)             # [n_probes * n_atoms]

                src_list.append(src)
                dst_list.append(dst)
        
            edge_src = torch.cat(src_list, dim=0)
            edge_dst = torch.cat(dst_list, dim=0)
            edge_index_PA = torch.stack([edge_src, edge_dst], dim=0)
            edge_index_AP = edge_index_PA[[1,0]] #PA -> AP
            
            rel_vec_AP = (frac_coords_probes[edge_index_AP[1]] - frac_coords_atoms[edge_index_AP[0]]) % 1.
            rel_vec_AP = rel_vec_AP - (rel_vec_AP > 0.5).float()
            
            rel_vec_PA = (frac_coords_atoms[edge_index_PA[1]] - frac_coords_probes[edge_index_PA[0]]) % 1.
            rel_vec_PA = rel_vec_PA - (rel_vec_PA > 0.5).float()
        
            return edge_index_AP, rel_vec_AP, edge_index_PA, rel_vec_PA

        # kNN (recommended)
        elif self.edge_style_charge == 'knn':
            # In the current setting, probe coordinates are fixed and do not change
            # Therefore, KNN apporach can be applied in an exteremly efficient manner
            # In this case, for each atom node, edges are formed with k=27 nearest probes (in fractional coordinate)

            edge_index = graph_between_node_probe(frac_coords_atoms, num_atoms, self.chgden_resol)

            edge_index_AP = edge_index
            edge_index_PA = edge_index[[1,0]] #AP -> PA
            
            rel_vec_AP = (frac_coords_probes[edge_index_AP[1]] - frac_coords_atoms[edge_index_AP[0]]) % 1.
            rel_vec_AP = rel_vec_AP - (rel_vec_AP > 0.5).float()
            
            rel_vec_PA = (frac_coords_atoms[edge_index_PA[1]] - frac_coords_probes[edge_index_PA[0]]) % 1.
            rel_vec_PA = rel_vec_PA - (rel_vec_PA > 0.5).float()
            
            return edge_index_AP, rel_vec_AP, edge_index_PA, rel_vec_PA
        
        else:
            raise Exception("Unexpected graph type")
    
 
            
    def forward(self, t, atom_types, frac_coords, lattices, charge_dens, num_atoms, node2graph):
        
        # Assign probe coordinates. 
        frac_coords_probe = self.frac_coords_probe.to(frac_coords.device) # (512,)
        
        num_probes = torch.full((num_atoms.shape[0],), self.chgden_resol**3).to(num_atoms.device)
        frac_coords_probe = frac_coords_probe.repeat(num_atoms.shape[0],1) # (512 * B,)

        # Assign probe2graph [0, ... , 0, 1, ... , 1, ..., B-1, ..., B-1]  (the batch index to which each probe belongs)
        probe2graph = torch.arange(len(num_atoms)).repeat_interleave(self.chgden_resol**3) # (512 * B,)
        probe2graph = probe2graph.to(frac_coords.device)
                
        # prepare graph for atoms (edge : atom <-> atom)
        # edges are preparted to be symmetric (i,j) <-> (j,i)
            # edges_AA : [2, #edges]  ->  [[first atom indices][second atom indices]] 
            # frac_diff_AA : [#edges, 3] ;  vector of each edges      
            # edge2graph_AA : [#edges] ; Index of the graph which each edge belongs   
                                
        edges_AA, frac_diff_AA = self.gen_edges_AA(num_atoms, frac_coords)
        edge2graph_AA = node2graph[edges_AA[0]]  # represents the batch index to which each edge belongs
        
        # prepare graph for probes (edge : atom -> probe)
        # edges_AP and edges_PA are flipped versions of each other
            # edges_AP : [2, #edges]  ->  [[atom indices][probe indices]]
            # frac_diff_AP : [#edges, 3] ; vector of each edges (A->P)           
            # edge2graph_AP : [#edges] ; Index of the graph which each edge belongs   
            
            # edges_PA : [2, #edges]  ->  [[probe indices][atom indices]]
            # frac_diff_PA : [#edges, 3] ; vector of each edges (P->A)             
            # edge2graph_PA : [#edges] ; Index of the graph which each edge belongs   
            
        edges_AP, frac_diff_AP, edges_PA, frac_diff_PA = self.gen_edges_PA(num_atoms, num_probes, frac_coords, frac_coords_probe)

        # represents the batch index to which each edge belongs
        edge2graph_AP = node2graph[edges_AP[0]]
        edge2graph_PA = node2graph[edges_PA[1]]

                    
        graph_dict = {
            'edge_index_AA' : edges_AA,
            'edge_graph_AA' : edge2graph_AA,
            'frac_diff_AA'  : frac_diff_AA,
            
            'edge_index_AP' : edges_AP, 
            'edge_graph_AP' : edge2graph_AP,
            'frac_diff_AP'  : frac_diff_AP,
            
            'edge_index_PA' : edges_PA, 
            'edge_graph_PA' : edge2graph_PA,
            'frac_diff_PA'  : frac_diff_PA,
        }

        #### Initialize Node Embeddings ####

        node_features = self.node_embedding(atom_types) # (num_atom, 512) ; num_atom = total number of atoms in all graphs

        t_per_atom = t.repeat_interleave(num_atoms, dim=0) # (num_atom, 256)
                
        node_features = torch.cat([node_features, t_per_atom], dim=1) # (num_atom, 768)
        
        node_features = self.atom_latent_emb(node_features) # (num_atom, 512)
        


        #### Initialize Probe Embeddings ####
    
        charge_dens_expand = self.charge_tot_latent_emb(charge_dens) # (B,256,8,8,8)
            
        charge_dens_expand = charge_dens_expand.permute(0,2,3,4,1).reshape(-1, self.expansion_dim) #(B x 512, 256)

        t_per_probe = t.repeat_interleave(self.chgden_resol**3, dim=0) 

        probe_features = torch.cat([charge_dens_expand, t_per_probe], dim=1)
        probe_features = self.probe_latent_emb(probe_features)  # (B x 512, 512)
                                                            
                      
                                                       
        #### Joint Processing ####
        for i in range(0, self.num_layers):
            node_features, probe_features = self._modules["csp_layer_%d" % i](node_features, probe_features, lattices, graph_dict)

        if self.ln:
            node_features = self.final_layer_norm_A(node_features)
            probe_features = self.final_layer_norm_P(probe_features)


        #### Readout Structure Components and Charge Density ####

        # Lattice (L)
        graph_features_node = scatter(node_features, node2graph, dim = 0, reduce = 'mean') # (B, 512)
        
        graph_features_probe = scatter(probe_features, probe2graph, dim = 0, reduce = 'mean') # (B, 512)
        
        graph_features = torch.cat([graph_features_node, graph_features_probe], dim=1)
        
        lattice_out = self.lattice_out(graph_features)
        
        lattice_out = lattice_out.view(-1, 3, 3)
        
        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
        
        
        # Charge Density (C)
        charge_out = self.charge_out(probe_features) # [B x 512, 3]
                         
        # conver back to voxel form
        charge_out = charge_out.view(-1, self.chgden_resol, self.chgden_resol, self.chgden_resol, self.chgden_dim)
        charge_out = charge_out.permute(0, 4, 1, 2, 3)
                
        
        # Node feature update (add global probe information for X and A)
        probe_graph_feature_per_atom = graph_features_probe[node2graph]  # [N_atoms, d]
        node_features = torch.cat([node_features, probe_graph_feature_per_atom], dim=1)
             
        # Coordinate (X)
        coord_out = self.coord_out(node_features)
             
        # Atom Type (A)
        if self.pred_type:
            type_out = self.type_out(node_features)
            
            return  type_out, coord_out, lattice_out, charge_out

        # For CSP task (not considered in this work)
        return coord_out, lattice_out, charge_out
