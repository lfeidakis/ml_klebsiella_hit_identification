# Leveraging Predictive AMR Models for Hit Identification in *Klebsiella pneumoniae*

![Pipeline Overview](pipeline.png)

This repository implements a focused workflow for identifying potential antimicrobial hits against *Klebsiella pneumoniae*, leveraging components from the [MultimodalAMR](https://github.com/BorgwardtLab/MultimodalAMR) project. Only the code necessary to run and extend the ResMLP classifier model is included. Please refer to the original repository for full architecture details, dataset preparation, and package requirements.

Candidate compounds are selected from the [Enamine REAL Hit Locator Library](https://enamine.net/compound-libraries/diversity-libraries/hit-locator-library-460), a commercially available compound library containing approximately 460,000 chemicals. The original `.sdf` file provided by Enamine was processed to generate two CSV files:

1. One file contains the molecular properties provided by the vendor as well as Morgan fingerprints (1024 bits) for each compound. Compounds with identical fingerprints were considered duplicates and removed.
2. The second file includes the same molecular properties and Morgan fingerprints, along with Molformer embeddings for each compound. Duplicate entries based on identical Molformer embeddings were also removed.

Both files are included in the compound list archive, which can be downloaded [here](https://drive.google.com/drive/folders/1BskzBQZoI-ezUXUrzfXGV7aHwV1Rp4Tm?usp=sharing). This folder must be added to the root folder of the project.

The full DRIAMS dataset and all necessary model checkpoints/configuration files can be downloaded from [here](https://drive.google.com/file/d/1ASajnwy5SnlfFVY_PO9MgOAPegCs72JM/view?usp=sharing). The dataset includes samples from all hospitals (A, B, C, and D) and all available years. The provided folder contains:  
- known (sample, drug, response) triplet outcomes  
- raw binned MALDI-TOF spectra  
- MAE embeddings of the spectra  
- the full train/validation/test split  
- model checkpoints and configuration files for two setups: `ResMLP_raw_morgan` and `ResMLP_MAE_Molformer`  
  
After downloading, the contents should be placed in the root directory of the project for everything to run as expected.
 

The following five Jupyter notebooks form the core of the pipeline:
- `sample_prefiltering`: Selects *K. pneumoniae* test samples from the dataset and applies two preselection filters based on model performance and known resistance profiles.
- `resistance_pred`: Uses a pretrained ResMLP model to predict resistance probabilities for vendor compounds.
- `compound_ranking`: Aggregates predictions across test samples and ranks compounds using two scoring methods.
- `chemical_filtering`: Applies standard medicinal chemistry filters (Lipinski, Veber, PAINS) and structural dissimilarity checks.
- `supplementary`: Provides model diagnostics, training curves, and performance metrics (e.g. MCC).

Each notebook contains a detailed explanation of what its individual cells do. To reproduce the results of the study, the notebooks must be run in the order presented above, and the cells within each notebook must be executed sequentially.

Two training scripts are also included:
- `finetune_ResMLP_classifier.py`: Fine-tunes the ResMLP classifier on *K. pneumoniae* samples.
- `train_ResAMR_classifier_non_balanced_loss_weights.py`: Trains a ResMLP model from scratch using custom, unbalanced loss weights.

**Note to the reader:**  
The code in this repository was written and executed on macOS. If you are using a different operating system (e.g. Windows or Linux), you may need to make minor adjustments — particularly with respect to file paths or OS-specific behaviors.