import sys
sys.path.insert(0, "..")
sys.path.insert(0, "./")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../data_split")

import numpy as np
import os
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

from data_split.data_utils import DataSplitter
from models.data_loaders_newest import DrugResistanceDataset_Fingerprints, SampleEmbDataset, DrugResistanceDataset_Embeddings
from models.classifier import Residual_AMR_Classifier
import sys

def main(args):

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    checkpoint = torch.load(args.ckpt_path, weights_only=True, map_location="mps")

    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}

    model = Residual_AMR_Classifier(config)
    model.load_state_dict(state_dict)
    model.train()

    print("Pretrained model loaded successfully")

    if model.spectrum_emb.in_features == 2048:
        sample_spectra_type = "MAE"
    else :
        sample_spectra_type = "rawSpectra"


    seed = args.seed

    # Setup output folder to save results
    finetune_folder = f"finetune/ResMLP_{sample_spectra_type}_{args.drug_emb_type}_{args.fingerprint_class}_pretrained_on:{args.pretrain_dataset}_finetuned_on:{args.finetune_dataset}_for_{args.n_epochs}_epochs_weight_decay:{args.weight_decay}_lr:{args.learning_rate}"
    if not exists(finetune_folder):
        os.makedirs(finetune_folder, exist_ok=True)

    # Read data
    driams_long_table = pd.read_csv(args.driams_long_table)
    if args.target_species is not None:
        driams_long_table = driams_long_table[(driams_long_table['species'] == args.target_species)]
    spectra_matrix = np.load(args.spectra_matrix).astype(float)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    split_idsList=args.split_ids
    dataset_investigated="any"
    
    # Instantate data split
    dsplit = DataSplitter(driams_long_table, dataset="any")
    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    

    train_df, val_df, test_df = dsplit.specific_train_val_test_split(split_idsList, dataset_investigated, driams_long_table)

    test_df.to_csv(join(finetune_folder, "test_set.csv"), index=False)

    if args.drug_emb_type=="fingerprint":
        train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)
        val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)
        test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)
    elif args.drug_emb_type=="vae_embedding" or args.drug_emb_type=="gnn_embedding":
        train_dset = DrugResistanceDataset_Embeddings(train_df, spectra_matrix, drugs_df, samples_list)
        val_dset = DrugResistanceDataset_Embeddings(val_df, spectra_matrix, drugs_df, samples_list)
        test_dset = DrugResistanceDataset_Embeddings(test_df, spectra_matrix, drugs_df, samples_list)


    # Save configuration
    if not exists(join(finetune_folder, "config.json")):
        with open(join(finetune_folder, "config.json"), "w") as f:
            json.dump(config, f)


    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    # Instantiate pytorch lightning experiment
    experiment = Classifier_Experiment(config, model)

    # Save summary of the model architecture
    if not exists(join(finetune_folder, "architecture.txt")):
        with open(join(finetune_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())


    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(finetune_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(finetune_folder, "logs/"))

    # Finetune model
    print("Finetuning..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", 
        default_root_dir=finetune_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                          logger=tb_logger, log_every_n_steps= 1
                        # limit_train_batches=6, limit_val_batches=4, limit_test_batches=4
                         )
    

    metrics = {"train_loss": [], "val_loss": [], "mcc_val": []}

    class MetricLogger(pl.Callback):

        def on_train_epoch_end(self, trainer, pl_module):
            if "train_loss" in trainer.callback_metrics:
                train_loss = trainer.callback_metrics["train_loss"].item()
                metrics["train_loss"].append(train_loss)

        def on_validation_epoch_end(self, trainer, pl_module):
            if "val_loss" in trainer.callback_metrics:
                val_loss = trainer.callback_metrics["val_loss"].item()
                metrics["val_loss"].append(val_loss)
                
            if "val_mcc" in trainer.callback_metrics:  
                mcc_val = trainer.callback_metrics["val_mcc"].item()
                metrics["mcc_val"].append(mcc_val)

    trainer.callbacks.append(MetricLogger())

    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Finetuning complete!")

    with open(join(finetune_folder, f"metrics_seed{seed}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Test model
    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(finetune_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)
    
    experiment.model.eval()
    test_df["Predictions"] = experiment.test_predictions
    test_df.to_csv(join(finetune_folder, f"test_set_seed{seed}.csv"), index=False)
  
    
    print("Testing complete")



if __name__=="__main__":

    parser = ArgumentParser()
    
    parser.add_argument("--split_ids", type=str,
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/data_splits.csv")

    parser.add_argument("--training_setup", type=int)
    
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_long_table", type=str, 
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/maskedAE_copy10_batch50_embSize512_epoch100_MR25_data.npy")
    parser.add_argument("--drugs_df", type=str, 
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/drug_fingerprints_Mol_selfies.csv" )

    parser.add_argument("--drug_emb_type", type=str, default="fingerprint", choices=["fingerprint", "vae_embedding", "gnn_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="molformer_github", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem", "none", "molformer_huggingFace", "molformer_github", "selfies_label", "selfies_flattened_one_hot"])


    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.8) #0.003
    parser.add_argument("--weight_decay", type=float, default=1e-5) #1e-5

    parser.add_argument("--target_species", type=str, default="Klebsiella pneumoniae")

    parser.add_argument("--ckpt_path", type=str, 
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/ResultsAndCheckpoints/ABCD/MAE_Mol/new_loader_MAE_Mol_ABCD_DRIAMS-any_specific/0/lightning_logs/version_0/checkpoints/epoch=99-step=394700.ckpt")
    parser.add_argument("--config_path", type=str,
                        default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/ResultsAndCheckpoints/ABCD/MAE_Mol/new_loader_MAE_Mol_ABCD_DRIAMS-any_specific/config.json")

    parser.add_argument("--pretrain_dataset", type=str, choices=['A2018', 'B2018', 'C2018', 'D2018', 'A18BCD', 'ABCD' ], default="ABCD")
    parser.add_argument("--finetune_dataset", type=str, choices=['A2018', 'B2018', 'C2018', 'D2018', 'A18BCD', 'ABCD' ], default="ABCD")

    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    args.species_embedding_dim = 0


    main(args)