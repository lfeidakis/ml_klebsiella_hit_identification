import sys

sys.path.insert(0, "..")
#sys.path.insert(0, "../processed_data/")
sys.path.insert(0, "./")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../data_split")
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from os.path import join, exists
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from argparse import ArgumentParser
import json
from experiments.pl_experiment import Classifier_Experiment
import itertools
import pubchempy as pcp

from data_split.data_utils import DataSplitter
from models.data_loaders import (DrugResistanceDataset_Fingerprints, SampleEmbDataset, DrugResistanceDataset_Embeddings,
                                 DrugResistanceDataset_Smiles)
from models.classifier import Residual_AMR_Classifier
import sys

import shap

# Molformer
from urllib.request import urlopen
from urllib.parse import quote
import yaml
from argparse import Namespace
#from fast_transformers.masking import LengthMask as LM
sys.path.append('../molformer-main/notebooks/pretrained_molformer/')
#from train_pubchem_light import LightningModule
from tokenizer.tokenizer import MolTranBertTokenizer
from sklearn.metrics import roc_auc_score
#from rdkit import Chem
from sklearn.linear_model import LogisticRegression
############################

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i + batch_size, len(data))]
        i += batch_size


#def embed(model, smiles, tokenizer, batch_size=64):
#    model.eval()
#    embeddings = []
#    for batch in batch_split(smiles, batch_size=batch_size):
#        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
#        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
#        with torch.no_grad():
#            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
#        # average pooling over tokens
#        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#        embedding = sum_embeddings / sum_mask
#        embeddings.append(embedding.detach().cpu())
#    return torch.cat(embeddings)


def tok_emb(smiles, tokenizer):
    batch_enc = tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
    idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
    return idx, mask


#def canonicalize(s):
#    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)


def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'


def PCPconvert(ids):
    try:
        results = pcp.get_compounds(ids, 'name')
        return results[0].isomeric_smiles
    except:
        return 'Did not work'


TRAINING_SETUPS = list(
    itertools.product(['A', 'B', 'C', 'D'], ["random", "drug_species_zero_shot", "drugs_zero_shot"], np.arange(5)))


def main(args):

    """
    this script allows for the training of the ResAMR classifier from fingerprints, molformer embedding,
    molformer tokens, and other embeddings
    :param args: model, data and training parameters
    :return: saves model test results
    """

    config = vars(args)
    seed = args.seed
    # Setup output folders to save results
    output_folder = join(args.output_folder, args.experiment_group, args.experiment_name, str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    results_folder = join(args.output_folder, args.experiment_group, args.experiment_name + "_results")
    if not exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    experiment_folder = join(args.output_folder, args.experiment_group, args.experiment_name)
    if exists(join(results_folder, f"test_metrics_{args.seed}.json")):
        sys.exit(0)
    if not exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)

    # Read processed_data
    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix).astype(float)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    split_idsList=args.split_ids
    dataset_investigated=args.driams_dataset

    ####################################################################################################################
    # Code to use Molformer Encoding
    ####################################################################################################################

    # Convert to smiles
    if exists(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_{args.driams_dataset}.csv')):
        drugs_df_smiles = pd.read_csv(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_{args.driams_dataset}.csv'))
    else:
        smiles = []
        for i in drugs_df.index:
            smiles.append(PCPconvert(i))  # old function: CIRconvert(i))
        drugs_df_smiles = pd.DataFrame(data={'smiles': smiles}, index=drugs_df.index)
        drugs_df_smiles.to_csv(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_{args.driams_dataset}.csv'))

    if args.drug_emb_type == 'molformer_smiles_embedding':
        if exists(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_embedding_{args.driams_dataset}.csv')):
            print('load embedding')
            smiles_embedd = np.genfromtxt(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_embedding_{args.driams_dataset}.csv'), delimiter=",").astype('float32')
            drugs_df_smiles_embedd = pd.DataFrame(data={'smiles_embedding': list(smiles_embedd)}, index=drugs_df.index)
        #else:
        #    # load tokenizer
        #    tokenizer = MolTranBertTokenizer(args.molformer_bert)
        #    # print(tokenizer.vocab)
        #    # load embedding model
        #    with open(args.molformer_config, 'r') as f:
        #        config_emb = Namespace(**yaml.safe_load(f))
        #    # print(config_emb)

            # load molformer
        #    ckpt = args.molformer_checkpoint
        #    cwd = os.getcwd()
        #    os.chdir(args.molformer_directory)
        #    lm = LightningModule(config_emb, tokenizer.vocab).load_from_checkpoint(ckpt, config=config_emb, vocab=tokenizer.vocab)
        #    print(lm)
        #    os.chdir(cwd)

        #    # embedd
        #    smiles = drugs_df_smiles.smiles.apply(canonicalize)
        #    smiles_embedd = embed(lm, smiles, tokenizer).numpy()
        #    drugs_df_smiles_embedd = pd.DataFrame(data={'smiles_embedding': list(smiles_embedd)}, index=drugs_df.index)

            # save smiles embedding
        #    np.savetxt(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_embedding_{args.driams_dataset}.csv'),
        #               smiles_embedd, delimiter=",")

    elif args.drug_emb_type == 'tokenizer_smiles_embedding':
        if exists(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_tokenizer_{args.driams_dataset}.csv')):
            print('load embedding')
            smiles_idx = np.genfromtxt(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_tokenizer_{args.driams_dataset}.csv'), delimiter=",").astype('float32')
            drugs_df_smiles_embedd = pd.DataFrame(data={'smiles_embedding': list(list(smiles_idx))}, index=drugs_df.index) # double list ?
        #else:
        #    # load tokenizer
        #    tokenizer = MolTranBertTokenizer(args.molformer_bert)
        #    # print(tokenizer.vocab)
        #    # embedd
        #    smiles = drugs_df_smiles.smiles.apply(canonicalize)
        #    idx, _ = tok_emb(smiles, tokenizer)
        #    smiles_idx = idx.float().numpy()
        #    drugs_df_smiles_embedd = pd.DataFrame(data={'smiles_embedding': list(list(smiles_idx))}, index=drugs_df.index)

            # save smiles embedding
        #    np.savetxt(os.path.join(os.path.dirname(args.drugs_df), f'drug_smiles_tokenizer_{args.driams_dataset}.csv'),
        #               smiles_idx, delimiter=",")

    ####################################################################################################################
    ####################################################################################################################

    # Instantate processed_data split
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    # Split selection for the different experiments.
    if args.split_type == "random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2,
                                                                       random_state=args.seed)
    elif args.split_type == "drug_species_zero_shot":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2,
                                                                   random_state=args.seed)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    elif args.split_type == "drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if args.seed >= len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[args.seed]
        # target_drug = args.drug_name
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)

    ### Added Diane
    elif args.split_type == "specific":
        train_df, val_df, test_df = dsplit.specific_train_val_test_split(split_idsList, dataset_investigated, driams_long_table)
    ###

    test_df.to_csv(join(output_folder, "test_set.csv"), index=False)

    if args.drug_emb_type == "fingerprint":
        train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list,
                                                        fingerprint_class=config["fingerprint_class"])
        val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list,
                                                      fingerprint_class=config["fingerprint_class"])
        test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list,
                                                       fingerprint_class=config["fingerprint_class"])
    elif args.drug_emb_type == "vae_embedding" or args.drug_emb_type == "gnn_embedding":
        train_dset = DrugResistanceDataset_Embeddings(train_df, spectra_matrix, drugs_df, samples_list)
        val_dset = DrugResistanceDataset_Embeddings(val_df, spectra_matrix, drugs_df, samples_list)
        test_dset = DrugResistanceDataset_Embeddings(test_df, spectra_matrix, drugs_df, samples_list)
    ####################################################################################################################
    # Code to use Molformer Encoding
    ####################################################################################################################
    elif "smiles" in args.drug_emb_type:
        train_dset = DrugResistanceDataset_Smiles(train_df, spectra_matrix, drugs_df_smiles_embedd, samples_list)
        val_dset = DrugResistanceDataset_Smiles(val_df, spectra_matrix, drugs_df_smiles_embedd, samples_list)
        test_dset = DrugResistanceDataset_Smiles(test_df, spectra_matrix, drugs_df_smiles_embedd, samples_list)
    ####################################################################################################################
    ####################################################################################################################

    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)
    del config["seed"]
    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)
    if not exists(join(results_folder, "config.json")):
        with open(join(results_folder, "config.json"), "w") as f:
            json.dump(config, f)

    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Instantiate model and pytorch lightning experiment
    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    # Save summary of the model architecture
    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())
    if not exists(join(results_folder, "architecture.txt")):
        with open(join(results_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())

    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=join(output_folder, "logs/"))

    # Train model
    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto",  # "auto"
                         default_root_dir=output_folder, max_epochs=args.n_epochs,  # , callbacks=callbacks,
                         #  logger=tb_logger, log_every_n_steps=3
                         # limit_train_batches=6, limit_val_batches=4, limit_test_batches=4
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Training complete!")

    # Test model
    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(results_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)

    background_loader = DataLoader(test_dset, batch_size=100, shuffle=True)
    background = next(iter(background_loader))
    X_background = torch.cat(background[:3], dim=1)

    test_fi_loader = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)
    fi_batch = next(iter(test_fi_loader))
    X_fi_batch = torch.cat(fi_batch[:3], dim=1)

    experiment.model.eval()
    test_df["Predictions"] = experiment.test_predictions
    test_df.to_csv(join(results_folder, f"test_set_seed{seed}.csv"), index=False)

    if args.eval_importance:
        print("Evaluating feature importance")

        class ShapWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, X):
                species_idx = X[:, 0:1]
                x_spectrum = X[:, 1:6001]
                dr_tensor = X[:, 6001:]
                response = []
                dataset = []
                batch = [species_idx, x_spectrum, dr_tensor, response, dataset]
                return experiment.model(batch)

        shap_wrapper = ShapWrapper(experiment.model)
        explainer = shap.DeepExplainer(shap_wrapper, X_background)
        shap_values = explainer.shap_values(X_fi_batch)
        column_names = ["Species"] + [f"Spectrum_{i}" for i in range(6000)] + [f"Fprint_{i}" for i in range(1024)]
        np.save(join(results_folder, f"shap_values_seed{seed}.npy"), shap_values)
        if not exists(join(output_folder, "shap_values_columns.json")):
            with open(join(output_folder, "shap_values_columns.json"), "w") as f:
                json.dump(column_names, f, indent=2)

        shap_values_df = pd.DataFrame(shap_values, columns=column_names)
        shap_values_df.to_csv(join(output_folder, "shap_values.csv"), index=False)

    print("Testing complete")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="myExperiment")
    parser.add_argument("--experiment_group", type=str, default="ResMLP_smiles_datasplit") # _newspectra")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--split_type", type=str, default="specific",
                        choices=["random", "drug_species_zero_shot", "drugs_zero_shot", "specific"])
    parser.add_argument("--split_ids", type=str, default="D:\\AICenterProject08\\ai_center_project_8\\data_split\\data_splits.csv")

    ###
    parser.add_argument("--training_setup", type=int)
    parser.add_argument("--eval_importance", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D', 'any'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="F:\\AICenterProject08\\data\\spectra_binned_6000_2018.npy ")
                        # default="F:\\AICenterProject08\\data\\encoded_DRIAMSB_split1_MA_copy10_batch50_embSize512_epoch100_MR25.npy")
    parser.add_argument("--drugs_df", type=str,
                        # default="../processed_data/GNN_embeddings.csv")
                        default="../processed_data/drug_fingerprints.csv")

    # molformer ###################
    parser.add_argument("--molformer_bert", type=str,
                        default='D:\\AICenterProject08\\ai_center_project_8\\molformer-main\\notebooks\\pretrained_molformer\\bert_vocab.txt')
    parser.add_argument("--molformer_checkpoint", type=str,
                        default="D:\\AICenterProject08\\ai_center_project_8\\molformer-main\\data\\Pretrained_MoLFormer\\checkpoints\\N-Step-Checkpoint_3_30000.ckpt")
    parser.add_argument("--molformer_directory", type=str,
                        default="D:\\AICenterProject08\\ai_center_project_8\\molformer-main\\notebooks\\pretrained_molformer")
    parser.add_argument("--molformer_config", type=str,
                        default="D:\\AICenterProject08\\ai_center_project_8\\molformer-main\\data\\Pretrained_MoLFormer\\hparams.yaml")
    ###############################

    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=6000)  # 512)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)

    parser.add_argument("--drug_emb_type", type=str, default="molformer_smiles_embedding",
                        choices=["fingerprint", "vae_embedding", "gnn_embedding", 'tokenizer_smiles_embedding',
                                 'molformer_smiles_embedding'])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024",
                        choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem", "none"])
    # changed to 768 for molformer embedding # changed to 236 for molformer tokens
    parser.add_argument("--fingerprint_size", type=int, choices=[128, 236, 768], default=768)

    parser.add_argument("--n_hidden_layers", type=int, default=8)  # 5

    parser.add_argument("--n_epochs", type=int, default=500)  # 500
    parser.add_argument("--batch_size", type=int, default=256)  # 128
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.0003)  # 0.003
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()

    # modify args
    if args.training_setup is not None:
        dataset, split_type, seed = TRAINING_SETUPS[args.training_setup]

        args.seed = seed
        args.driams_dataset = dataset
        args.split_type = split_type
    args.species_embedding_dim = 0

    for layers in [5]:
        args.n_hidden_layers = layers
        args.experiment_name = args.experiment_name + f"_DRIAMS-{args.driams_dataset}_{args.split_type}_{layers}"

        main(args)
