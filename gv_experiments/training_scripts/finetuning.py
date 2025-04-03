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
from tqdm import tqdm
from argparse import ArgumentParser
import json

from data_split.data_utils import DataSplitter
from models.data_loaders_newest import DrugResistanceDataset_Fingerprints, DrugResistanceDataset_Embeddings
from models.classifier import Residual_AMR_Classifier

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc='Training'):
        species_idx, spectrum, fprint_tensor, response, dataset = batch

        species_idx = species_idx.to(device)
        spectrum = spectrum.to(device)
        fprint_tensor = fprint_tensor.to(device)
        response = response.to(device).unsqueeze(-1)

        optimizer.zero_grad()
        outputs = model((species_idx, spectrum, fprint_tensor, response, dataset))
        loss = criterion(outputs, response)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            species_idx, spectrum, fprint_tensor, response, dataset = batch

            species_idx = species_idx.to(device)
            spectrum = spectrum.to(device)
            fprint_tensor = fprint_tensor.to(device)
            response = response.to(device).unsqueeze(-1)

            outputs = model((species_idx, spectrum, fprint_tensor, response, dataset))
            loss = criterion(outputs, response)
            running_loss += loss.item()

    return running_loss / len(val_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            species_idx, spectrum, fprint_tensor, response, dataset = batch

            species_idx = species_idx.to(device)
            spectrum = spectrum.to(device)
            fprint_tensor = fprint_tensor.to(device)
            response = response.to(device).unsqueeze(-1)

            outputs = model((species_idx, spectrum, fprint_tensor, response, dataset))
            loss = criterion(outputs, response)
            running_loss += loss.item()

            probas = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probas)
            actuals.extend(response.cpu().numpy().flatten())

    return running_loss / len(test_loader), predictions, actuals

def main(args):
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    checkpoint = torch.load(args.ckpt_path, weights_only=True, map_location="mps")
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}

    model = Residual_AMR_Classifier(config)
    model.load_state_dict(state_dict)
    model.train()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    sample_spectra_type = "MAE" if model.spectrum_emb.in_features == 2048 else "rawSpectra"

    finetune_folder = f"ResMLP_{sample_spectra_type}_fingerprint_{args.fingerprint_class}_pretrained_on:{args.pretrain_dataset}_finetuned_on:{args.finetune_dataset}_for_{args.n_epochs}_epochs"
    os.makedirs(finetune_folder, exist_ok=True)

    config_path = join(finetune_folder, "config.json")
    architecture_path = join(finetune_folder, "architecture.txt")
    model_checkpoint_path = join(finetune_folder, "best_model.pth")
    test_set_path = join(finetune_folder, "test_set.csv")
    test_predictions_path = join(finetune_folder, "test_predictions.csv")
    metrics_path = join(finetune_folder, "metrics.json")

    with open(config_path, "w") as f:
        json.dump(config, f)

    with open(architecture_path, "w") as f:
        f.write(str(model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    driams_long_table = pd.read_csv(args.driams_long_table)
    if args.target_species:
        driams_long_table = driams_long_table[driams_long_table['species'] == args.target_species]

    spectra_matrix = np.load(args.spectra_matrix).astype(float)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)

    dsplit = DataSplitter(driams_long_table, dataset="any")
    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    train_df, val_df, test_df = dsplit.specific_train_val_test_split(args.split_ids, "any", driams_long_table)
    test_df.to_csv(test_set_path, index=False)

    train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)
    val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)
    test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=args.fingerprint_class)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(args.n_epochs):
        print(f"\nEpoch {epoch + 1}/{args.n_epochs}")

        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_checkpoint_path)
            print('Model checkpoint saved.')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('Early stopping triggered.')
                break

    test_loss, predictions, actuals = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    test_df["Predictions"] = predictions
    test_df.to_csv(test_predictions_path, index=False)

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--split_ids", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/data_splits.csv")
    parser.add_argument("--driams_long_table", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/rawSpectra_data.npy")
    parser.add_argument("--drugs_df", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/drug_fingerprints_Mol_selfies.csv")
    parser.add_argument("--ckpt_path", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/ResultsAndCheckpoints/ABCD/raw_fing/new_loader_rawMS_fing_ABCD_DRIAMS-any_specific/0/lightning_logs/version_0/checkpoints/epoch=99-step=394700.ckpt")
    parser.add_argument("--config_path", type=str, default="/Users/lfeidakis/Desktop/MultimodalAMR-main-updated/ABCD/ResultsAndCheckpoints/ABCD/raw_fing/new_loader_rawMS_fing_ABCD_DRIAMS-any_specific/config.json")
    parser.add_argument('--target_species', type=str, default="Klebsiella pneumoniae")
    parser.add_argument('--fingerprint_class', type=str, default='morgan_1024')

    parser.add_argument("--pretrain_dataset", type=str, default="ABCD")
    parser.add_argument("--finetune_dataset", type=str, default="ABCD")

    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())

    args = parser.parse_args()
    main(args)

