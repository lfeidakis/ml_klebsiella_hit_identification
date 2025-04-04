# Leveraging Predictive Antimicrobial Resistance Models for Hit Identification in *Klebsiella pneumoniae*

This repository contains all code and analysis for a pipeline that leverages predictive antimicrobial resistance (AMR) models to identify potential novel antimicrobial candidates specifically targeting *Klebsiella pneumoniae*. The project builds on the foundation of the [MultimodalAMR](https://github.com/BorgwardtLab/MultimodalAMR) framework and introduces targeted post-processing, filtering, and evaluation logic tailored to this pathogen.

---

## 📁 Repository Contents

### 🔬 Original Contributions
This repository includes **five custom Jupyter notebooks** and **two training scripts** developed to extend and specialize the MultimodalAMR pipeline:

#### Jupyter Notebooks
1. `sample_prefiltering`:  
   Extracts *Klebsiella pneumoniae* test samples and applies two biological filters to reduce noise from unreliable samples.

2. `resistance_pred`:  
   Runs resistance prediction models on vendor compound data and generates per-sample probability files.

3. `compound_ranking`:  
   Aggregates predictions across test samples to produce ranked lists of promising compounds using multiple scoring strategies.

4. `chemical_filtering`:  
   Applies chemical filters (Lipinski, Veber, PAINS) and computes dissimilarity to known antibiotics.

5. `supplementary`:  
   Provides additional visualizations, training curves, and MCC evaluation logic for model performance analysis.

#### Training Scripts
- `finetune_ResMLP_classifier.py`: Fine-tunes the ResMLP classifier on selected training data.  
- `train_ResAMR_classifier_non_balanced_loss_weights.py`: Trains the ResAMR classifier using custom, non-balanced loss weighting.

---

## 🚀 Getting Started

### Installation
Set up a conda environment using the base `MultimodalAMR` repo instructions:  
[https://github.com/BorgwardtLab/MultimodalAMR](https://github.com/BorgwardtLab/MultimodalAMR)

Make sure to install:
- `rdkit`
- `pytorch`
- `tqdm`
- `scikit-learn`
- `seaborn`

### Usage
1. Follow the MultimodalAMR data preparation pipeline to generate embeddings and training splits.  
2. Run the notebooks in the following order for analysis:
   - `sample_prefiltering`
   - `resistance_pred`
   - `compound_ranking`
   - `chemical_filtering`
   - `supplementary`

3. For training custom models, use the provided training scripts with your desired configuration.

---

## 📖 Attribution

This repository builds heavily on code from the original [MultimodalAMR repository](https://github.com/BorgwardtLab/MultimodalAMR) by BorgwardtLab. Most core model architectures, dataset formats, and utility functions are preserved from their original work.

Please refer to their repository for citations, licensing, and in-depth explanations of model design.

---

## 🧠 Acknowledgements

- Original research framework: **MultimodalAMR**  
- This extension was developed for research focused on *Klebsiella pneumoniae*-specific antimicrobial hit identification.

---

## 📄 License

See the original license in the [MultimodalAMR repository](https://github.com/BorgwardtLab/MultimodalAMR). This work is provided under the same terms unless otherwise stated.
