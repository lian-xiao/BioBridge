from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
cfg_path = "configs/CDAN.yaml"
data = "bindingdb"
comet_support = False

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")
set_seed(cfg.SOLVER.SEED)
mkdir(cfg.RESULT.OUTPUT_DIR)
experiment = None
print(f"Config yaml: {cfg_path}")
print(f"Running on: {device}")
print(f"Hyperparameters:")
dict(cfg)
dataFolder = f'./datasets/{data}/cluster'

train_source_path = os.path.join(dataFolder, 'source_train.csv')
train_target_path = os.path.join(dataFolder, 'target_train.csv')
test_target_path = os.path.join(dataFolder, 'target_test.csv')
df_train_source = pd.read_csv(train_source_path)
df_train_target = pd.read_csv(train_target_path)
df_test_target = pd.read_csv(test_target_path)

train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func}
training_generator = DataLoader(train_dataset, **params)
params['shuffle'] = False
params['drop_last'] = False
source_generator = DataLoader(train_dataset, **params)
target_generator = DataLoader(train_target_dataset, **params)
n_batches = max(len(source_generator), len(target_generator))
multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
val_generator = DataLoader(test_target_dataset, **params)
test_generator = DataLoader(test_target_dataset, **params)


model = DrugBAN(**cfg).to(device)
if cfg.DA.USE:
    if cfg["DA"]["RANDOM_LAYER"]:
        domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
    else:
        domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                   n_class=cfg["DECODER"]["BINARY"]).to(device)
    # params = list(model.parameters()) + list(domain_dmm.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True
trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm,
                          experiment=experiment, **cfg)
result = trainer.test()
with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
    wf.write(str(model))
print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")