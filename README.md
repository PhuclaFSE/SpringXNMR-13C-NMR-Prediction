\# 🧪 SpringXNMR: 13C NMR Prediction with GATv2



\*\*SpringXNMR\*\* is an AI-powered web application designed to predict \*\*$^{13}$C NMR chemical shifts\*\* for organic molecules. Instead of relying on traditional lookup tables or expensive DFT calculations, this tool utilizes \*\*Graph Attention Networks (GATv2)\*\* to understand the local and global electronic environments of carbon atoms within a molecular graph.



\## 🚀 Key Features

\- \*\*AI-Driven Prediction:\*\* Predict $\\delta$ (ppm) values instantly from a SMILES string.

\- \*\*Graph Attention Mechanism:\*\* The GATv2 architecture identifies relevant neighboring functional groups to optimize prediction accuracy.

\- \*\*Structural Visualization:\*\* Real-time 2D molecular structure rendering upon input.

\- \*\*Data Export:\*\* Download prediction results as a `.csv` file for further research and documentation.



\## 🛠️ Installation \& Setup



1\.  \*\*Clone the repository:\*\*

&#x20;   ```bash

&#x20;   git clone \[https://github.com/your-username/SpringXNMR-13C-NMR-Prediction.git](https://github.com/your-username/SpringXNMR-13C-NMR-Prediction.git)

&#x20;   cd SpringXNMR-13C-NMR-Prediction

&#x20;   ```



2\.  \*\*Install dependencies:\*\*

&#x20;   ```bash

&#x20;   pip install -r requirements.txt

&#x20;   ```



3\.  \*\*Launch the application:\*\*

&#x20;   ```bash

&#x20;   streamlit run app/app.py

&#x20;   ```



\## 🏗️ Architecture \& Methodology

The project is built using the \*\*PyTorch Geometric\*\* framework. Molecules are represented as graphs where:

\- \*\*Nodes:\*\* Represent atoms (features include atomic number, electronegativity, hybridization, aromaticity, etc.).

\- \*\*Edges:\*\* Represent chemical bonds (features include bond type, conjugation, and stereochemistry).

\- \*\*Core Model:\*\* A deep Graph Neural Network using GATv2 layers, which allows the model to learn dynamic weights for interactions between atoms, effectively capturing the shielding and deshielding effects.



\---



\## ⚠️ Current Limitations

While the model provides high-speed inference, it still has some following limits:

\- \*\*Element Coverage:\*\* The model is optimized for common organic elements(C, H, N, O, P, S, F, Cl, Br, I). Predictions for organometallics or rare elements may be less stable.

\- \*\*Solvent Effects:\*\* Currently, the model predicts based on the isolated molecular structure and does not take into account the changes of chemical shift due to specific solvents (e.g., $CDCl\_3$ vs. $DMSO-d\_6$).

\- \*\*Accuracy:\*\* The most significant current challenge is accuracy; predicted results may have an error margin of approximately 19-20 ppm compared to actual experimental chemical shifts.

\## 🌟 Future Improvement

SpringXNMR aims to evolve into a more comprehensive suite for computational chemistry:

\- \*\*Hyperparameter \& Optimization\*\* The overall performance of the model is now considered one of the most important tasks. Future iterations expect to reduce the mean error to below 5 ppm, providing the near-exact values for scientific research

\- \*\*Dataset Expansion\*\* Adding a more diverse range of chemical environments and solvents to bridge the current accuracy gap 

\---



\*\*Author:\*\* Gia Phuc la  

\*\*Project:\*\* SpringXNMR Project - 2026  

\*\*License:\*\* MIT

