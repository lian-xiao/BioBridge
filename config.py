import argparse

# 定义 train_config 参数解析
def get_train_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--data_root', type=str, default='Data', help='Root directory for data')
    parser.add_argument('--dataset_name', type=str, default='bindingdb/cold/', help='Dataset name')
    parser.add_argument('--measure_name', type=str, nargs='+', default=['Y'], help='Measure names')
    parser.add_argument('--gamma', type=float, default=10, help='Gamma value')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--optimizer_momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=2.5e-5, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--main_metric', type=str, default='auroc', help='Main metric')
    parser.add_argument('--finetune_load_path', type=str, default=None, help='Path to load fine-tuned model')
    parser.add_argument('--Da_warm_epochs', type=int, default=0, help='Da warm-up epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints_allstem/', help='Checkpoints folder')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--train_dataset_length', type=int, default=None, help='Train dataset length')
    parser.add_argument('--test_dataset_length', type=int, default=None, help='Test dataset length')
    parser.add_argument('--eval_dataset_length', type=int, default=None, help='Eval dataset length')
    parser.add_argument('--DDP', action='store_true', help='Enable Distributed Data Parallel')
    return parser.parse_args()

# 定义 model_config 参数解析
def get_model_args():
    parser = argparse.ArgumentParser(description='Model configuration')
    parser.add_argument('--drugban', action='store_true', help='Enable drugban')
    parser.add_argument('--muti_out', action='store_true', help='Enable multi output')
    parser.add_argument('--p_ems2_emb', action='store_true', help='Enable p_ems2 embedding')
    parser.add_argument('--d_molformer_emb', action='store_true', help='Enable d_molformer embedding')
    parser.add_argument('--p_emb', type=int, default=128, help='P embedding size')
    parser.add_argument('--d_emb', type=int, default=128, help='D embedding size')
    parser.add_argument('--d_stem_channel', type=int, default=128, help='D stem channel size')
    parser.add_argument('--stem_kernel', type=int, default=1, help='Stem kernel size')
    parser.add_argument('--p_stem', action='store_true', help='Enable p_stem')
    parser.add_argument('--d_stem', action='store_true', help='Enable d_stem')
    parser.add_argument('--gate', action='store_true', help='Enable gate')
    parser.add_argument('--p_stem_channel', type=int, default=128, help='P stem channel size')
    parser.add_argument('--d_channels', type=int, nargs='+', default=[128, 128, 128], help='D channels')
    parser.add_argument('--p_channels', type=int, nargs='+', default=[128, 128, 128], help='P channels')
    parser.add_argument('--d_out_channel', type=int, default=128, help='D output channel size')
    parser.add_argument('--p_out_channel', type=int, default=128, help='P output channel size')
    parser.add_argument('--out_hidden_size', type=int, default=256, help='Output hidden size')
    parser.add_argument('--layers_num', type=int, default=3, help='Number of layers')
    parser.add_argument('--binary', type=int, default=1, help='Binary flag')
    return parser.parse_args()

if __name__ == '__main__':
    train_args = get_train_args()
    model_args = get_model_args()
    print('Training Configuration:', train_args)
    print('Model Configuration:', model_args)