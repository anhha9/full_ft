import os
import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from transformers.optimization import AdamW
import evaluate
from datasets import load_dataset, Dataset
import torch
import wandb

# Initialize wandb
wandb.init(project="full_fine_tune")  # Replace "your_project_name" with your actual project name

parser = argparse.ArgumentParser(description='Train or continue training T5 model.')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to save or load model checkpoints')
parser.add_argument('--train_from_scratch', action='store_true', help='Flag to train model from scratch')
args = parser.parse_args()

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

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Load the dataset
dataset = load_dataset("csv", data_files="/work/tc062/tc062/haanh/full_ft/no_long_token.csv")
df = pd.read_csv('/work/tc062/tc062/haanh/full_ft/no_long_token.csv', encoding="utf-8", delimiter=',')
dataset = Dataset.from_pandas(df[['sentences', 'normalizations']])
train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1111).values()

# Ensure val_dataset is sliced correctly to get the first 1000 examples
val_dataset = val_dataset.select(range(1000))

def preprocess_data(examples):
    inputs = examples['sentences']
    targets = examples['normalizations']
    inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')
    labels["input_ids"] = [[label_id if label_id != tokenizer.pad_token_id else -100 for label_id in label_ids] for
                           label_ids in labels["input_ids"]]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }

train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=2,
    load_best_model_at_end=True,
    report_to="wandb",  # Enable wandb
    gradient_checkpointing=True
)

# Set PyTorch memory management settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train(checkpoint)

# Evaluate the model on the validation dataset
metrics = trainer.evaluate(val_dataset)  # <--- Added evaluation step
accuracy_score = metrics["eval_accuracy"]  # <--- Extracted accuracy score

# Print the accuracy score
print(f"My model is done training and the accuracy score is {accuracy_score:.4f}")

# Save the final model and tokenizer
output_dir = args.checkpoint_dir if args.checkpoint_dir else "/work/tc062/tc062/haanh/full_ft/final_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
