import re

from sklearn.model_selection import train_test_split
from torch import nn

from my_model import Dti_cnn_mutiout, Dti_cnn

comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
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

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', default='configs/CADA.yaml', help="path to config file", type=str)
parser.add_argument('--model', default='dti_multiout', help="model type", choices=['drugban', 'dti_multiout', 'dti_multi'])
parser.add_argument('--data', default='bindingdb/', type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='meta_unseen_protein', type=str, metavar='S', help="split task"), #choices=['random', 'cold', 'cluster','unseen_drug'])
parser.add_argument('--save', default='result/', type=str)
parser.add_argument('--seed',default=2026,type=int)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(args.seed)
    suffix = str(int(time() * 1000))[6:]
    current_directory = os.path.join(args.save,args.model,args.data,args.split)

    if not os.path.exists(current_directory):
        mkdir(current_directory)
    # 获取当前目录下所有文件夹的名字
    folders = [folder for folder in os.listdir(current_directory) if
               os.path.isdir(os.path.join(current_directory, folder))]

    # 使用正则表达式匹配以"version"开头的文件夹，并提取版本数字
    version_pattern = re.compile(r'^version(\d+)$')
    matching_folders = [folder for folder in folders if version_pattern.match(folder)]
    version_num = [int(version_pattern.match(folder).group(1)) for folder in matching_folders]
    if len(version_num) == 0:
        version_num = 0
    else:
        version_num = max(version_num)+1
    version = 'version'+str(version_num)
    cfg.RESULT.OUTPUT_DIR = os.path.join(current_directory,version)

    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'../Data/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path).dropna()
        df_val = pd.read_csv(val_path).dropna()
        df_test = pd.read_csv(test_path).dropna()

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        # if 'meta' in dataFolder:
        #     train_source_path = os.path.join(dataFolder, 'source_train.csv')
        #     train_target_path = os.path.join(dataFolder, 'target_train.csv')
        #     df_train_source = pd.read_csv(train_source_path)
        #     df_train_target = pd.read_csv(train_target_path)
        #     df_train_target,df_test_target = train_test_split(df_train_target, test_size=0.2, random_state=42)
        # else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)
    if args.model == 'drugban':
        model = DrugBAN(**cfg).to(device)
    elif args.model == 'dti_multiout':
        model = Dti_cnn_mutiout(cfg).to(device)
    elif args.model == 'dti_multi':
        model = Dti_cnn(cfg).to(device)
    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            if cfg["DA"]["MuDa"]:
                domain_dmm = nn.ModuleList(
                    [nn.Sequential(Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"])).to(device) for _ in range(2)])
            else:
                domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)

        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
    else:
        trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm,
                          experiment=experiment, **cfg)
    result = trainer.train()
    # ckpt = os.path.join('result/madti_cada/',args.data,args.split)
    # result = trainer.test('test',ckpt)
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
