import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoConfig,
    ViTFeatureExtractor, 
    AutoModelForImageClassification, 
    Trainer, 
    TrainingArguments,
)

model_args = "google/vit-base-patch16-224-in21k"

train_data = load_dataset("cifar10", split='train[:50]')
val_data = load_dataset("cifar10", split='test[:50]')
labels = train_data.features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

config = AutoConfig.from_pretrained(
        model_args,
        num_labels=len(labels),
        # i2label=id2label,
        # label2id=label2id,
        # finetuning_task="image-classification",
)
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    cache_dir = None,
    revision = "main",
    use_auth_token = None,
)

# Define torchvision transforms to be applied to each image.
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
    [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)
_val_transforms = Compose(
    [
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ]
)

train_data = train_data.map(
    lambda examples: {'pixel_values': [_train_transforms(pil_img.convert("RGB")) for pil_img in train_data["img"]]}, batched=True
)
val_data = val_data.map(
    lambda examples: {'pixel_values': [_val_transforms(pil_img.convert("RGB")) for pil_img in val_data["img"]]}, batched=True
)


training_args = TrainingArguments(
    output_dir = "./test/test_results",
    gradient_accumulation_steps = 1,
    per_device_train_batch_size = 8,
    num_train_epochs = 5,
    fp16 = False,
)

def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_data, 
    eval_dataset=val_data, 
    data_collator=collate_fn,
)

print("### Start Training ###")
trainer.train()
print("### Save The Model ###")
trainer.save_model()
print("### Predict ###")
trainer.predict(test_dataset=val_data)
