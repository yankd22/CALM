import torch
from heads import get_classification_head
from tqdm import tqdm
from datasets.common import maybe_dictionarize, get_dataloader_shuffle
from datetime import datetime
import os


class MaskMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(MaskMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.args = args
        self.exam_datasets = exam_datasets

        self.frozen = ["model.model.positional_embedding", "model.model.text_projection", "model.model.logit_scale", "model.model.token_embedding.weight", 
        "model.model.ln_final.weight", "model.model.ln_final.bias"]
        
        self.trainable_paramslist = []
        for parameters in paramslist:
            trainable_parameters = []
            for idx, name in enumerate(self.names):
                if name not in self.frozen:
                    trainable_parameters.append(parameters[idx])
            self.trainable_paramslist.append(trainable_parameters)

        self.trainable_name = []
        for idx, name in enumerate(self.names):
            if name not in self.frozen:
                self.trainable_name.append(name)

        self.num_params = sum([p.numel() for p in self.trainable_paramslist[-1]])

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

        self.mask = self.create_basepatch()

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def set_attr(self, obj, names, val):
        if len(names) == 1:
            setattr(obj, names[0], val)
        else:
            self.set_attr(getattr(obj, names[0]), names[1:], val)

    def load_weights_grad(self, mod, names, params):
        for name, p in zip(names, params):
            self.set_attr(mod, name.split("."), torch.nn.Parameter(p, requires_grad=True))

    def create_basepatch(self):
        threshold = int(self.args.sparsity * self.num_params)

        abs_tv = []
        for p in self.trainable_paramslist[-1]: 
            abs_tv.append(torch.abs(p).view(-1)) 
        abs_tv = torch.cat(abs_tv)
        k = int(self.args.sparsity * abs_tv.numel())
        values, indices = torch.topk(abs_tv, k)
        threshold = values.min()  
        basepatch = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_paramslist[-1]]
        
        for p, q in zip(self.trainable_paramslist[-1], basepatch): 
            q[torch.abs(p) > threshold] = self.args.sigmoid_bias
            q[torch.abs(p) <= threshold] = -self.args.sigmoid_bias
        
        total_selected = sum([torch.sum(torch.round(torch.nn.Sigmoid()(p))) for p in basepatch])
        self.args.log.info(f'Total parameters selected in basepatch: {total_selected} / {self.num_params}')

        return basepatch


    def interpolate_model(self, round_, return_mask=False):
        sigmoid = torch.nn.Sigmoid()
        n_graft_params, n_total_params = 0, 0
        binary_mask = []
        paramlist_curr = []
        mask_idx = 0
        for idx, name in enumerate(self.names):
            # 从 paramslist 中获取预训练和微调参数
            pretrained_params = self.paramslist[0][idx]  # 预训练参数
            task_vector1 = self.paramslist[1][idx]
            task_vector2 = self.paramslist[-1][idx]  # 微调参数
            with torch.no_grad():
                if name in self.frozen:
                    paramlist_curr.append(pretrained_params + 0.5*(task_vector1 + task_vector2))
                else:
                    frac = sigmoid(self.mask[mask_idx])
                    mask_idx += 1
                    if round_:
                        frac = torch.round(frac)  # 二值化掩码
                        binary_mask.append(frac) 

                    interpolated_params = pretrained_params + (1-frac)*task_vector1 + frac * task_vector2

                    n_graft_params += torch.sum(frac)
                    n_total_params += pretrained_params.numel()
                    paramlist_curr.append(interpolated_params.to(self.device))

        self.load_weights_grad(self.model, self.names, paramlist_curr)

        if round_:
            self.args.log.info(f'Proportion in my graft: {n_graft_params / n_total_params}')

        if return_mask:
            return binary_mask, n_graft_params / n_total_params


    def save_binary_mask(self, datasets_dic,  exam_datasets, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        sigmoid = torch.nn.Sigmoid()
        binary_mask = []
        mask_idx = 0
        
        # 生成 binary_mask
        for idx, name in enumerate(self.names):
            if name not in self.frozen:
                frac = sigmoid(self.mask[mask_idx])
                mask_idx += 1
                frac = torch.round(frac)
                binary_mask.append(frac.cpu())  # 移动到 CPU 并添加到列表
        
        dataset_ids = []
        for dataset_group in exam_datasets:
            group_ids = ''.join([str(datasets_dic[dataset]) for dataset in dataset_group if dataset in datasets_dic])
            dataset_ids.append(group_ids)
        
        # 使用 '_' 连接不同组别的文件名
        file_name = f"mask{'_'.join(dataset_ids)}.pth"
        # 保存 binary_mask 到指定路径
        full_file_path = os.path.join(file_path, file_name)
        
        torch.save(binary_mask, full_file_path)
        
        print(f"Binary mask saved to {full_file_path}")

    

    def load_binary_mask(self, datasets_dic, exam_datasets, folder_path):
        dataset_ids = []
        for dataset_group in exam_datasets:
            group_ids = ''.join([str(datasets_dic[dataset]) for dataset in dataset_group if dataset in datasets_dic])
            dataset_ids.append(group_ids)

        file_name = f"mask_{'_'.join(dataset_ids)}.pth"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            return 0

        self.binary_mask = torch.load(file_path)
        mask_idx = 0
        paramlist_curr = []

        for idx, name in enumerate(self.names):
            pretrained_params = self.paramslist[0][idx]
            task_vector1 = self.paramslist[1][idx]
            task_vector2 = self.paramslist[-1][idx]

            with torch.no_grad():
                if name in self.frozen:
                    paramlist_curr.append(pretrained_params + 0.5 * (task_vector1 + task_vector2))
                else:
                    frac = self.binary_mask[mask_idx]
                    mask_idx += 1
                    interpolated_params = pretrained_params + (1 - frac) * task_vector1 + frac * task_vector2
                    paramlist_curr.append(interpolated_params.to(self.device))

        self.load_weights_grad(self.model, self.names, paramlist_curr)

        print(f"Binary mask loaded and applied from {file_path}")

        return 1


    def interpolate_init_model(self): 
        paramlist_curr = []
        for idx, name in enumerate(self.names):
            pretrained_params = self.paramslist[0][idx]
            task_vector1 = self.paramslist[1][idx]
            with torch.no_grad():
                if name in self.frozen:
                    paramlist_curr.append(pretrained_params + task_vector1)
                else:

                    interpolated_params = pretrained_params + task_vector1
                    paramlist_curr.append(interpolated_params.to(self.device))

        self.load_weights_grad(self.model, self.names, paramlist_curr)


    def train(self, seen_dataset, seen_dataname):
        criterion = torch.nn.CrossEntropyLoss()
        sigmoid = torch.nn.Sigmoid()

        lr = self.args.learning_rate
        total_grad = []
        first_batch = 0
        self.interpolate_model(round_=False)
        for i, dataset_name in enumerate(seen_dataname):
            layer_name = 'classifier_{}'.format(dataset_name)
            classification_head = getattr(self, layer_name)
            classification_head.to(self.device)
            dataloader = get_dataloader_shuffle(seen_dataset[i])
            for idx, data in enumerate(tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(self.device)
                y = data['labels'].to(self.device)
                features = self.model(x)
                outputs = classification_head(features)
                loss = criterion(outputs, y)
                # loss = criterion(outputs, y)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if n in self.trainable_name :
                        if p.grad is None: print (n)
                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters() if n in self.trainable_name]
                self.model.zero_grad()
                grad = [-g * task1.to(self.device) + g * task2.to(self.device)
                    for (g, task1, task2) in zip(grad, self.trainable_paramslist[1], self.trainable_paramslist[-1])]
                if first_batch == 0:
                    total_grad = [lr * g.detach().cpu() for g in grad]
                    first_batch += 1
                else:
                    total_grad = [p + lr * g.detach().cpu() for (p, g) in zip(total_grad, grad)]

                if idx > 0:
                    break

        total_grad = [p / (1. * first_batch) for p in total_grad]
        with torch.no_grad():
            for p, g in zip(self.mask, total_grad):
                derivative = sigmoid(p) * (1 - sigmoid(p))
                reg_term = self.args.l1_strength * torch.where(p > 0, derivative, -derivative)
                p -= g * derivative - reg_term


    def get_params(self, loaded=False):
        paramlist_curr = []
        sigmoid = torch.nn.Sigmoid()
        mask_idx = 0
        for idx, name in enumerate(self.names):
            task_vector1 = self.paramslist[1][idx]
            task_vector2 = self.paramslist[-1][idx]
            with torch.no_grad():
                if name in self.frozen:
                    paramlist_curr.append(0.5*(task_vector1 + task_vector2))
                else:
                    if not loaded:
                        frac = sigmoid(self.mask[mask_idx])
                        frac = torch.round(frac)
                    else:
                        frac = self.binary_mask[mask_idx]
                    mask_idx += 1
                    interpolated_params = (1-frac)*task_vector1 + frac * task_vector2
                    interpolated_params = interpolated_params.detach().requires_grad_().cpu()
                    paramlist_curr.append(interpolated_params)
        return tuple(paramlist_curr)

    def eval_dataset(self, seen_dataloader, seen_dataname, loaded=False):
        if not loaded:
            self.interpolate_out_model(round_=True)
        metrics = {}
        for i, dataset_name in enumerate(seen_dataname):
            layer_name = 'classifier_{}'.format(dataset_name)
            classification_head = getattr(self, layer_name)
            classification_head.to(self.device)
            self.model.eval()
            with torch.no_grad():
                top1, correct, n = 0., 0., 0.
                for idx, data in enumerate(tqdm(seen_dataloader[i])):
                    data = maybe_dictionarize(data)
                    x = data['images'].to(self.device)
                    y = data['labels'].to(self.device)
                    features = self.model(x)
                    outputs = classification_head(features)
                    pred = outputs.argmax(dim=1, keepdim=True).to(self.device)

                    correct += pred.eq(y.view_as(pred)).sum().item()
                    
                    n += y.size(0)

                top1 = correct / n
            metrics[dataset_name] =  top1
            current_time = datetime.now().strftime('%H:%M:%S')
            self.args.log.info(f'[{current_time}] Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
        
        return metrics
    
