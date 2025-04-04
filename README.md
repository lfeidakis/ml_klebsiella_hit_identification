# Leveraging Predictive AMR Models for Hit Identification in *Klebsiella pneumoniae*

This repository implements a focused antimicrobial candidate identification pipeline for *Klebsiella pneumoniae*, building on components from the [MultimodalAMR](https://github.com/BorgwardtLab/MultimodalAMR) project. It includes only the code necessary to run and extend the ResMLP classifier model, along with newly developed modules for sample filtering, resistance prediction, compound ranking, and chemical post-processing.

The following five Jupyter notebooks form the core of the pipeline:
- `sample_prefiltering`: Extracts relevant test samples and applies biological filters to exclude unreliable cases.
- `resistance_pred`: Uses a pretrained ResMLP model to predict resistance probabilities for vendor compounds.
- `compound_ranking`: Aggregates predictions across test samples and ranks compounds using several scoring methods.
- `chemical_filtering`: Applies standard medicinal chemistry filters (Lipinski, Veber, PAINS) and structural dissimilarity checks.
- `supplementary`: Provides model diagnostics, training curves, and performance metrics (e.g. MCC).

Two training scripts are also included:
- `finetune_ResMLP_classifier.py`: Fine-tunes the ResMLP classifier.
- `train_ResAMR_classifier_non_balanced_loss_weights.py`: Trains a ResAMR model using custom, unbalanced loss weights.

Only the code required to use the ResMLP classifier is included from the original MultimodalAMR project. Please refer to their repository for full architecture details, dataset preparation, and broader multimodal support.

