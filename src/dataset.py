import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem

from torch_geometric import edge_index
from numpy import dtype
class GetObjFeature:
  def __init__(self, path):
    self.path = path
    self.df = pd.read_pickle(path)
  def _getnode(self, mol):
    electron_neg = {
        1: 2.20,  # H
        6: 2.55,  # C
        7: 3.04,  # N
        8: 3.44,  # O
        9: 3.98,  # F
        15: 2.19, # P
        16: 2.58, # S
        17: 3.16, # Cl
        35: 2.96, # Br
        53: 2.66  # I
    }
    node = []
    for atom in mol.GetAtoms():
      atomic_num = atom.GetAtomicNum()
      num_hs = atom.GetTotalNumHs()
      degree = atom.GetDegree()
      is_aromatic = atom.GetIsAromatic()
      formal_charge = atom.GetFormalCharge()
      hyb = atom.GetHybridization()
      hyb_onehot = [
          1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
          1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
          1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
      ]
      electron_negative = electron_neg.get(atom.GetAtomicNum(), 2)
      is_in_ring = atom.IsInRing()
      total_valence = atom.GetTotalValence()
      is_ring5 = float(atom.IsInRingSize(5))
      is_ring6 = float(atom.IsInRingSize(6))

      vector = [atomic_num, num_hs, degree, is_aromatic, formal_charge, electron_negative, is_in_ring, total_valence] + hyb_onehot + [is_ring5, is_ring6]
      node.append(vector)
    return torch.tensor(data=node, dtype=torch.float)
  def _getbond(self, mol):
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
      i = bond.GetBeginAtomIdx()
      j = bond.GetEndAtomIdx()
      is_conjugated = float(bond.GetIsConjugated())
      bond_type = bond.GetBondType()
      one_hot_bond_type = [
          1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
          1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
          1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
          1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
      ]
      is_in_ring = bond.IsInRing()
      stereo = bond.GetStereo()
      one_hot_stereo = [
          1.0 if stereo == Chem.rdchem.BondStereo.STEREONONE else 0.0,
          1.0 if stereo == Chem.rdchem.BondStereo.STEREOE else 0.0,
          1.0 if stereo == Chem.rdchem.BondStereo.STEREOZ else 0.0,
          1.0 if stereo == Chem.rdchem.BondStereo.STEREOANY else 0.0
      ]
      vector = [is_conjugated] + one_hot_bond_type + one_hot_stereo + [is_in_ring]
      edge_indices.append([i, j])
      edge_attrs.append(vector)
      edge_indices.append([j, i])
      edge_attrs.append(vector)

    edge_index = torch.tensor(data=edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(data=edge_attrs, dtype=torch.float)
    return edge_index, edge_attr
  def _getitem(self, mol_obj, target_y):
    node = self._getnode(mol_obj)
    edge_index, edge_attr = self._getbond(mol_obj)

    y = [val if val is not None else float('nan') for val in target_y]
    y_tensor = torch.tensor(data=y, dtype=torch.float)
    carbon_mask = (node[:, 0] == 6.0)

    data = Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, carbon_mask=carbon_mask)
    return data
  def run(self):
    dataset = []
    for index, row in self.df.iterrows():
      mol_obj = row['Mol_obj']
      target_y = row['Target_Y']
      graph_data = self._getitem(mol_obj=mol_obj, target_y=target_y)
      dataset.append(graph_data)
    return dataset