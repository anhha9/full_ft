import resource

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory  # KiB

resource.setrlimit(resource.RLIMIT_AS,(int(get_memory() * 1024 / 5), (int(get_memory() * 1024 / 4)))

import os
import argparse
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch
import evaluate
import psutil  # Import psutil for memory monitoring
import tracemalloc

# Function to log memory usage
def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_memory_mb = mem_info.rss / (1024 ** 2)  # Convert bytes to MB
    print(f"Memory usage at {stage}: {rss_memory_mb:.2f} MB")


parser = argparse.ArgumentParser(description='Train or continue training T5 model.')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to save or load model checkpoints')
parser.add_argument('--train_from_scratch', action='store_true', help='Flag to train model from scratch')
args = parser.parse_args()

# Start tracemalloc
tracemalloc.start()

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

if args.train_from_scratch:
    print("Training from scratch")
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
else:
    checkpoint = args.checkpoint_dir if args.checkpoint_dir else "/work/tc062/tc062/haanh/full_ft/checkpoints"
    if not os.path.exists(checkpoint) or not os.path.isfile(os.path.join(checkpoint, "config.json")):
        print(f"Checkpoint directory {checkpoint} does not contain a valid model. Training from scratch instead.")
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
    else:
        print(f"Training from checkpoint: {checkpoint}")
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)


# Log memory usage before dataset loading
log_memory_usage("before dataset loading")

# Load the dataset
dataset = load_dataset("csv", data_files="/work/tc062/tc062/haanh/full_ft/no_long_token.csv")
df = pd.read_csv('/work/tc062/tc062/haanh/full_ft/no_long_token.csv', encoding="utf-8", delimiter=',')
dataset = Dataset.from_pandas(df[['sentences', 'normalizations']])
train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1111).values()

# Select 1000 examples for validation
val_dataset = val_dataset.select(range(1000))

# Log memory usage after dataset loading
log_memory_usage("after dataset loading")


# Define custom dataset class
class MyDataset(TorchDataset):
    """
    Custom Dataset class to handle tokenization and formatting for FLAN-T5 model.
    """

    def __init__(self, hf_dataset, tokenizer):
        self.examples = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        inputs = example['sentences']
        targets = example['normalizations']
        inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
        labels = self.tokenizer(targets, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


# Create DataLoader
train_dataloader = DataLoader(MyDataset(train_dataset, tokenizer), batch_size=4, shuffle=True)
val_dataloader = DataLoader(MyDataset(val_dataset, tokenizer), batch_size=4)
test_dataloader = DataLoader(MyDataset(test_dataset, tokenizer), batch_size=4)

# Log memory usage after DataLoader creation
log_memory_usage("after DataLoader creation")

# Define evaluation metric
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(pred):
    print("Starting eval")
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0].argmax(axis=-1)
    else:
        preds = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return accuracy_metric.compute(predictions=preds, references=labels)


# Define training arguments
training_args = TrainingArguments(
    output_dir=args.checkpoint_dir,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=5e-5,
    max_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=2,
    load_best_model_at_end=True,
    report_to="none",
    logging_dir='./logs',
    logging_steps=1000,
)

# Initialize the Trainer with custom DataLoader
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset,
    eval_dataset=val_dataloader.dataset,
    compute_metrics=compute_metrics,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data]),
    },
)

# Set PyTorch memory management settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Log memory usage before training starts
log_memory_usage("before training starts")

# Take initial memory snapshot
snapshot1 = tracemalloc.take_snapshot()

# Log memory usage after training
log_memory_usage("after training")

# Take final memory snapshot
snapshot2 = tracemalloc.take_snapshot()

# Compare snapshots and print top differences
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

# print("[ Top 10 differences ]")
for stat in top_stats[:10]:
    print(stat)

# Evaluate the model on the validation dataset
metrics = trainer.evaluate(val_dataloader.dataset)
accuracy_score = metrics["eval_accuracy"]

# Print the accuracy score
print(f"My model is done training and the accuracy score is {accuracy_score:.4f}")

# Save the final model and tokenizer
output_dir = args.checkpoint_dir if args.checkpoint_dir else "/work/tc062/tc062/haanh/full_ft/final_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Log memory usage after saving the model
log_memory_usage("after saving model")