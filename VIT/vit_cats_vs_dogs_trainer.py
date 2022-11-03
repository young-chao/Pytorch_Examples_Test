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
    ViTFeatureExtractor, 
    AutoModelForImageClassification, 
    Trainer, 
    TrainingArguments,
)

model_args = "google/vit-base-patch16-224-in21k"

dataset = load_dataset("hf-internal-testing/cats_vs_dogs_sample")["train"]
dataset = dataset.train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]


model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
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

dataset["train"] = dataset["train"].map(
    lambda examples: {'pixel_values': [_train_transforms(pil_img.convert("RGB")) for pil_img in dataset["train"]["image"]]}, batched=True
)
dataset["validation"] = dataset["validation"].map(
    lambda examples: {'pixel_values': [_val_transforms(pil_img.convert("RGB")) for pil_img in dataset["validation"]["image"]]}, batched=True
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
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset["train"], 
    eval_dataset=dataset["validation"], 
    data_collator=collate_fn,
)

print("### Start Training ###")
trainer.train()
print("### Save The Model ###")
trainer.save_model()
print("### Predict ###")
trainer.predict(test_dataset=dataset["validation"])
