# torchrun --nproc_per_node=2 vit_cifar10_ddp.py

import os
from sre_parse import expand_template
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from transformers import (
    AutoConfig,
    ViTFeatureExtractor, 
    AutoModelForImageClassification, 
)

# 参数定义
epochs = 5
lr = 2e-5
batch_size = 8
model_args = "google/vit-base-patch16-224-in21k"
dataset_args = "cifar10"

# 分布式训练初始化
dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)


# 数据集加载
train_data = load_dataset(dataset_args, split='train[:50]')
labels = train_data.features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# 模型加载
config = AutoConfig.from_pretrained(
        model_args,
        num_labels=len(labels),
        # i2label=id2label,
        # label2id=label2id,
        # finetuning_task="image-classification",
)
model = AutoModelForImageClassification.from_pretrained(model_args, config=config,)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_args, cache_dir = None, revision = "main", use_auth_token = None, )

# 数据预处理
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
    [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)
train_data = train_data.map(
    lambda examples: {'pixel_values': [_train_transforms(pil_img.convert("RGB")) for pil_img in train_data["img"]]}, batched=True
)
def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return pixel_values, labels  


train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=batch_size*size, collate_fn=collate_fn, sampler=train_sampler)
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=lr*size)

for epoch in range(1, epochs):
    total_loss = 0.0
    train_sampler.set_epoch(epoch)
    model.train()
    start_time = time.time()
    for step, batch_data in enumerate(train_loader):
        pixel_values, labels = batch_data
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(pixel_values=pixel_values, labels=labels)
        loss = output[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end_time = time.time()
    print(epoch, "training time:", end_time-start_time, "s")
