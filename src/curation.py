import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

class NMRDataCurator():
  def __init__(self, verbose=True, max_atoms=80):
    self.verbose = verbose
    self.max_atoms = max_atoms
    # RDKit tools
    self.lfc = rdMolStandardize.LargestFragmentChooser()
    self.uncharger = rdMolStandardize.Uncharger()
    self.normalizer = rdMolStandardize.Normalizer()
    # allowed atoms: H, B, C, N, O, F, Si, P, S, Cl, Br, I
    self.allowed_atoms = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

  def _Parse_Label(self, nmr_string):
    label = {}
    features = nmr_string.strip('|').split('|')
    for feature in features:
      parts = feature.split(';')
      if len(parts) >= 3:
        try:
          shift = float(parts[0])
          atom_idx = int(parts[-1]) - 1
          if atom_idx >= 0:
            label[atom_idx] = shift
          else: continue
        except ValueError: continue
    return label

  def _Assign_Label_To_Atoms(self, label, mol):
    for atom_idx, shift in label.items():
      if atom_idx < mol.GetNumAtoms():
        mol.GetAtomWithIdx(atom_idx).SetDoubleProp("NMR_SHIFT", shift)
    return mol

  def _Extract_Label_From_Atoms(self, mol):
    y = [None] * mol.GetNumAtoms()
    for i, atom in enumerate(mol.GetAtoms()):
      if atom.GetAtomicNum() == 6:
        if atom.HasProp("NMR_SHIFT"):
          y[i] = atom.GetDoubleProp("NMR_SHIFT")
    return y

  def _Validation_Structure(self, mol):
    try:
      Chem.SanitizeMol(mol)
      if mol.GetNumAtoms() > self.max_atoms: return None
      carbon = False
      for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in self.allowed_atoms: return None
        if atom.GetAtomicNum() == 6: carbon = True
        if atom.GetNumRadicalElectrons() > 0: return None
      return mol if carbon else None
    except Exception: return None

  def _Clean_Molecules(self, mol):
    frags = rdmolops.GetMolFrags(mol, asMols=True)
    if not frags:
      return mol
    mol = max(frags, key=lambda x: x.GetNumAtoms())
    mol = self.uncharger.uncharge(mol)
    return mol

  def _Normalize_Molecules(self, mol):
    mol = self.normalizer.normalize(mol)
    Chem.AssignStereochemistry(mol, force=True)
    return mol

  def _Is_Duplicate(self, mol, y):
    smiles = Chem.MolToSmiles(mol, canonical=True)
    shifts_tuple = tuple(round(val, 1) for val in y if val is not None)
    unique_key = (smiles, shifts_tuple)
    if unique_key in self.validated_smiles: return True
    self.validated_smiles.add(unique_key)
    return False

  def _Curation(self, mol):
    if mol is None: return None, None
    if mol.HasProp('Spectrum 13C 0'):
      nmr_string = mol.GetProp('Spectrum 13C 0')
      label = self._Parse_Label(nmr_string)
      if not label: return None, None
      mol = self._Assign_Label_To_Atoms(label, mol)
    else: return None, None
    mol = self._Validation_Structure(mol)
    if not mol: return None, None
    try:
      mol = self._Clean_Molecules(mol)
      mol = self._Normalize_Molecules(mol)
    except Exception: return None, None
    if not mol: return None, None
    y = self._Extract_Label_From_Atoms(mol)
    if all(val is None for val in y): return None, None
    if self._Is_Duplicate(mol, y): return None, None
    return mol, y

  def run(self, supplier):
    self.validated_smiles = set()
    results = []
    for mol in tqdm(supplier, disable=not self.verbose, desc="Curating Dataset"):
      curated_mol, y = self._Curation(mol)
      if curated_mol is not None:
        results.append((curated_mol, y))
    return results