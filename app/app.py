import sys
import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import pandas as pd
from torch_geometric.data import Data
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import NMRModel

class NMR_Parser:
    def __init__(self):
        self.electron_neg = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
            15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
        }

    def _getnode(self, mol):
        node_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            num_hs = atom.GetTotalNumHs()
            degree = atom.GetDegree()
            is_aromatic = float(atom.GetIsAromatic())
            formal_charge = float(atom.GetFormalCharge())
            hyb = atom.GetHybridization()
            hyb_onehot = [
                1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
                1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
                1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
            ]
            en = self.electron_neg.get(atomic_num, 2.0)
            is_in_ring = float(atom.IsInRing())
            total_valence = float(atom.GetTotalValence())
            is_ring5 = float(atom.IsInRingSize(5))
            is_ring6 = float(atom.IsInRingSize(6))

            vector = [atomic_num, num_hs, degree, is_aromatic, formal_charge, en, is_in_ring, total_valence] + hyb_onehot + [is_ring5, is_ring6]
            node_features.append(vector)
        
        return torch.tensor(node_features, dtype=torch.float)

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
            is_in_ring = float(bond.IsInRing())
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

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return edge_index, edge_attr

    def parse(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        node = self._getnode(mol)
        edge_index, edge_attr = self._getbond(mol)
        data = Data(x=node, edge_index=edge_index, edge_attr=edge_attr)
        return data, mol

parser = NMR_Parser()

st.set_page_config(page_title='NMR Predictor', layout='wide')
st.title('SPRINGX: 13C NMR Chemical Shift Predictor')
with st.container():
    st.markdown("""
    <div style="background-color:white; padding: 15px; border-radius: 10px; border-left: 5px solid red;">
        <h2 style="margin-top:0; color:#0e1117;">🌟 Welcome to SpringX NMR</h2>
        <p style="color:black">
            An intelligent 13C NMR prediction platform powered by <b>Graph Attention Networks</b>. 
            Feed your chemical structures into our <b>GATv2</b> model to receive instant chemical shift analysis.
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
@st.cache_resource
def load_assets():
    model = NMRModel()
    checkpoint = torch.load(r'D:\SpringXNMR_Project\SpringXNMR-13C-NMR-Prediction\data\best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    stats = joblib.load(r'D:\SpringXNMR_Project\SpringXNMR-13C-NMR-Prediction\data\nmr_stats.joblib') 
    return model, stats

model, stats = load_assets()

column_1, column_2 = st.columns([1,1])

with column_1:
    smiles = st.text_input('Type SMILES: ', placeholder='e.g. CC(=O)O')
    if st.button('Predict Now', type='primary'):
        if smiles:
            data, mol = parser.parse(smiles) 
            if data is not None:
                with torch.no_grad():
                    pred_norm = model(data).view(-1).numpy()
                    pred_ppm = pred_norm * stats['std'] + stats['mean']
                    
                    results = []
                    for i, atom in enumerate(mol.GetAtoms()):
                        if atom.GetAtomicNum() == 6: 
                            results.append({
                                'Atom Index': i, 
                                'Symbol': 'C', 
                                'Shift (ppm)': round(float(pred_ppm[i]), 2)
                            })
                    
                    st.session_state['results'] = results
                    st.session_state['mol'] = mol
            else:
                st.error('Invalid SMILES string!')
        else:
            st.warning('Please enter a SMILES string.')

with column_2:
    if 'mol' in st.session_state:
        st.subheader('2D Structure')
        img = Draw.MolToImage(st.session_state['mol'], size=(400, 400))
        st.image(img, use_container_width=False)

if 'results' in st.session_state:
    st.divider()
    st.subheader('Prediction table (13C)')
    df_res = pd.DataFrame(st.session_state['results'])
    st.dataframe(df_res, use_container_width=True)
    
    csv = df_res.to_csv(index=False).encode('utf-8')
    st.download_button('Download results (.csv)', data=csv, file_name='nmr_prediction.csv', mime='text/csv')