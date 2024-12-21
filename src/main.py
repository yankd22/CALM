import os

import time
from tqdm import tqdm
import sys

sys.path.append('/home/calm')
from datasets.common import get_dataloader, get_dataloader_shuffle
import torch
from task_vectors import TaskVector
from args import parse_arguments
from datasets.registry import get_dataset
from logits_extraction import credible_entropy_samples
from mask_merging import MaskMerging
import ast


# 建立日志
def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + '/' + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def create_log_with_mask_naming(datasets_dic, exam_datasets, args):
    # 生成基于 name_list 的文件名
    dataset_ids = []
    for dataset_group in exam_datasets:
        group_ids = ''.join([str(datasets_dic[dataset]) for dataset in dataset_group if dataset in datasets_dic])
        dataset_ids.append(group_ids)
    mask_name = '_'.join(dataset_ids)

    # 添加时间戳
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    # 生成日志文件名
    log_filename = 'log_{}_{}_calm_merging.txt'.format(mask_name, str_time_)

    # 创建日志
    log = create_log_dir(args.logs_path, log_filename)
    args.log = log


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# 获取task_num列表
name_list = ast.literal_eval(args.name_list)
model_name = 'ViT-B-32'
source_root_path = '/home/calm/'
args = parse_arguments()
args.data_location = source_root_path + 'dataset/'
args.model = model_name
args.save = source_root_path + model_name
args.binary_save = '/home/calm/binary_save/' + model_name
args.logs_path = '/home/calm/logs/' + model_name
args.sample_path = "/home/calm/samples/"
args.ratio = 0.3
args.standard = 0
args.sparsity = 1e-5
args.sigmoid_bias = 5
args.learning_rate = 1e7
args.num_train_epochs = 100
args.l1_strength = 1
pretrained_checkpoint = source_root_path + model_name + '/zeroshot.pt'

datasets_dic = {'SUN397': 1, 'Cars': 2, 'RESISC45': 3, 'EuroSAT': 4,
                'SVHN': 5, 'GTSRB': 6, 'MNIST': 7, 'DTD': 8}


def get_datasets(datasets_dic, name_list):
    datasets = []
    for sublist in name_list:
        converted_sublist = [key for index in sublist
                             for key, value in datasets_dic.items() if value == index]
        datasets.append(converted_sublist)
    return datasets


exam_datasets = get_datasets(datasets_dic, name_list)

create_log_with_mask_naming(datasets_dic, exam_datasets, args)


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features


pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model)
model = model.to(args.device)
_, names = make_functional(model)

args.log.info(f"datasets: {exam_datasets}")

best_param = []

for step in range(len(exam_datasets)):

    task_vectors = []
    task_vectors += [
        TaskVector(pretrained_checkpoint, source_root_path + model_name + '/' + dataset_name + '/finetuned.pt')
        for dataset_name in exam_datasets[step]
    ]

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]  # pretrains
    if step > 0:
        paramslist += [best_param[-1]]
    else:
        paramslist += []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for tv in
                   task_vectors]  # task vectors

    torch.cuda.empty_cache()

    seen_dataname = []
    for i in range(step + 1):
        for seen_name in exam_datasets[i]:
            seen_dataname.append(seen_name)

    if step == 0 and len(seen_dataname) > 2:
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist) - 1) * prior

        with torch.no_grad():
            param = params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, rlambdas[0]))) for _, p in
                                   enumerate(zip(*paramslist[1:])))
            best_param.append(param)

    else:
        dataset_current = exam_datasets[step][-1]
        mask_model = MaskMerging(paramslist, model, names, seen_dataname, args)

        seen_dataset = []
        for dataset_name in seen_dataname:
            credible_samples = credible_entropy_samples(args.sample_path, dataset_name, args.ratio, args.standard)
            dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location,
                                  credible_samples=credible_samples, batch_size=128)
            seen_dataset.append(dataset)

        test_seen_dataloader = []
        for dataset_name in seen_dataname:
            dataset_c = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location,
                                    batch_size=128)
            dataloader_c = get_dataloader(dataset_c, is_train=False, args=args, image_encoder=None)
            test_seen_dataloader.append(dataloader_c)

        for epoch in range(args.num_train_epochs):
            args.log.info(f'Current task: {dataset_current}, Epoch: {epoch}')
            mask_model.train(seen_dataset, seen_dataname)

        with torch.no_grad():
            param = mask_model.get_params(loaded=False)
            best_param.append(param)
            torch.cuda.empty_cache()

