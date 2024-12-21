from datasets import load_dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = load_dataset("SKNahin/bengali-transliteration-data")

# Load the tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['rm'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples['bn'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision if a GPU is available
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()