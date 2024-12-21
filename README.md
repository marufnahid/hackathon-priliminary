# Banglish-to-Bangla Transliteration

## Steps to Solve the Challenge

### 1. Load the Dataset
- The dataset is loaded from [Hugging Face](https://huggingface.co/datasets/SKNahin/bengali-transliteration-data).
- Split the dataset into training and validation subsets using an 80/20 split.

### 2. Data Preprocessing
- Tokenize both Banglish (`rm`) and Bangla (`bn`) text using `MT5Tokenizer`.
- Apply sequence-to-sequence tokenization with padding and truncation (max length: 128 tokens).
- Assign `input_ids` and `attention_mask` for inputs, and create labels for targets.

### 3. Select a Model
- The `google/mt5-small` model is chosen for fine-tuning.
- **Justification**:
  - Excellent for low-resource language tasks.
  - Multilingual capabilities make it suitable for Banglish-to-Bangla transliteration.
  - Efficient and lightweight (`small` variant).

### 4. Train the Model
- Fine-tune the model on the preprocessed dataset.
- Use the AdamW optimizer with the following hyperparameters:
  - **Learning Rate**: `2e-5`
  - **Batch Size**: `16`
  - **Epochs**: `3`
- Monitor training and validation loss to evaluate performance.
- Save the fine-tuned model for deployment.

