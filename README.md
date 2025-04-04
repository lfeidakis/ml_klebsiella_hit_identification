# Leveraging Predictive AMR Models for Hit Identification in *Klebsiella pneumoniae*

This repository implements a focused antimicrobial candidate identification pipeline for *Klebsiella pneumoniae*, building on components from the [MultimodalAMR](https://github.com/BorgwardtLab/MultimodalAMR) project, from which it includes the code necessary to run and extend the ResMLP classifier model. Please refer to their repository for full architecture details, dataset preparation, package requirements etc.

The following five Jupyter notebooks form the core of the pipeline:
- `sample_prefiltering`: Selects *K. pneumoniae* test samples from the dataset and applies two preselection filters based on model performance and known resistance profiles.
- `resistance_pred`: Uses a pretrained ResMLP model to predict resistance probabilities for vendor compounds.
- `compound_ranking`: Aggregates predictions across test samples and ranks compounds using two scoring methods.
- `chemical_filtering`: Applies standard medicinal chemistry filters (Lipinski, Veber, PAINS) and structural dissimilarity checks.
- `supplementary`: Provides model diagnostics, training curves, and performance metrics (e.g. MCC).

Each notebook contains a detailed explanation of what its individual cells do. To reproduce the results of the study, the notebooks must be run in the order presented above, and the cells within each notebook must be executed sequentially.

Two training scripts are also included:
- `finetune_ResMLP_classifier.py`: Fine-tunes the ResMLP classifier on *K. pneumoniae* samples.
- `train_ResAMR_classifier_non_balanced_loss_weights.py`: Trains a ResAMR model from scratch using custom, unbalanced loss weights.
