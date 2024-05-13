import random
import torch
import argparse
import numpy as np
import os

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from train import PMDP


'''
# Locking random seed
def seed_setting(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_setting()
'''

parser = argparse.ArgumentParser()
# dataset setting
parser.add_argument('--dataset', dest='dataset', default='WIKI', choices=['WIKI', 'IAPR', 'FLICKR', 'COCO', 'NUS'], help='name of the dataset')
parser.add_argument('--dataset_path', dest='dataset_path', default='../Datasets/', help='path of the dataset')
# retrieval model setting
parser.add_argument('--test_retrieval', dest='test_retrieval', action='store_true', help='to test retrieval model or not')
parser.add_argument('--retrieval_bit', dest='retrieval_bit', type=int, default=32, choices=[16, 32, 64, 128], help='bit length of the retrieval model')
parser.add_argument('--retrieval_method', dest='retrieval_method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN', 'DPSH', 'Bihalf'], help='name of the retrieval model')
parser.add_argument('--retrieval_models_path', dest='retrieval_models_path', default='./retrieval_models/', help='path of the retrieval model')
# evaluation setting
parser.add_argument('--map_k', dest='map_k', type=int, default=50)
parser.add_argument('--psr_k', dest='psr_k', type=int, default=10000)
# retrieval feedback awareness 1/2
parser.add_argument('--awareness', nargs='+', choices=['I2I', 'I2T', 'T2I', 'T2T'], help='awareness class')
parser.add_argument('--query_sample_number', dest='query_sample_number', type=int, default=2000)
parser.add_argument('--near_sample_number', dest='near_sample_number', type=int, default=5)
parser.add_argument('--rank_sample_number', dest='rank_sample_number', type=int, default=5)
# retrieval feedback awareness 2/2
parser.add_argument('--train_subnet', dest='train_subnet', action='store_true', help='to train subnet or not')
parser.add_argument('--test_subnet', dest='test_subnet', action='store_true', help='to test subnet or not')
parser.add_argument('--threshold_alpha', dest='threshold_alpha', type=float, default=0.1)
parser.add_argument('--parameter_alpha', dest='parameter_alpha', type=float, default=1.)
parser.add_argument('--parameter_beta', dest='parameter_beta', type=float, default=0.01)
parser.add_argument('--parameter_gamma', dest='parameter_gamma', type=float, default=0.001)
parser.add_argument('--subnet_bit', dest='subnet_bit', type=int, default=32)
parser.add_argument('--subnet_epoch', dest='subnet_epoch', type=int, default=50)
parser.add_argument('--subnet_batch_size', dest='subnet_batch_size', type=int, default=32)
parser.add_argument('--subnet_text_learning_rate', dest='subnet_text_learning_rate', type=float, default=1e-3)
parser.add_argument('--subnet_image_learning_rate', dest='subnet_image_learning_rate', type=float, default=1e-4)
# adversarial divergence diffusion
parser.add_argument('--train_diffusion', dest='train_diffusion', action='store_true', help='to train diffusion model or not')
parser.add_argument('--test_diffusion', dest='test_diffusion', action='store_true', help='to test diffusion model or not')
parser.add_argument('--threshold_beta', dest='threshold_beta', type=float, default=0.5)
parser.add_argument('--threshold_gamma', dest='threshold_gamma', type=float, default=-0.08)
parser.add_argument('--parameter_lambda', dest='parameter_lambda', type=float, default=10.)
parser.add_argument('--parameter_epsilon', dest='parameter_epsilon', type=float, default=5.)
parser.add_argument('--parameter_mu', dest='parameter_mu', type=float, default=5.)
parser.add_argument('--parameter_nu', dest='parameter_nu', type=float, default=5.)
parser.add_argument('--parameter_xi', dest='parameter_xi', type=float, default=100.)
parser.add_argument('--diffusion_epoch', dest='diffusion_epoch', type=int, default=50, help='number of epoch')
parser.add_argument('--diffusion_epoch_decay', dest='diffusion_epoch_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--diffusion_batch_size', dest='diffusion_batch_size', type=int, default=24, help='number of samples in one batch')
parser.add_argument('--diffusion_learning_rate', dest='diffusion_learning_rate', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--diffusion_learning_rate_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
# experiment detail
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
parser.add_argument('--output_path', dest='output_path', default='./outputs/', help='path of the output result')
parser.add_argument('--output_dir', dest='output_dir', default='output001', help='name of the output result')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

Dcfg = Dataset_Config(args.dataset, args.dataset_path)
X, Y, L = load_dataset(Dcfg.data_path)
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)

model = PMDP(args=args, Dcfg=Dcfg)

if args.test_retrieval:
    model.test_retrieval_model(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)

if args.train_subnet or args.train_diffusion:
    model.retrieval_feedback_awareness(Db_I, Db_T, Db_L)

if args.train_subnet:
    model.train_subnet(Db_I, Db_T, Db_L)

if args.test_subnet:
    model.test_subnet(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)

if args.train_diffusion:
    model.train_diffusion_model(Db_I, Db_T, Db_L)

if args.test_diffusion:
    model.test_diffusion_model(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)