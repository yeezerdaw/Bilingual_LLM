# Bilingual Tamil-English Language Model Fine-Tuning (Qwen1.5-1.8B)

**Author:** Balmuri Yeshwanth Kumar   
**Platform:** NVIDIA H100, CUDA 12.5
**Frameworks:** PyTorch, Hugging Face Transformers, PEFT, Datasets  

---

## Overview

This project focuses on building and fine-tuning a bilingual Tamil–English Large Language Model (LLM) using a curated combination of open-source text datasets. The objective is to develop a lightweight model capable of understanding and generating content fluently in both Tamil and English.

The model is fine-tuned using Low-Rank Adaptation (LoRA) techniques on the **Qwen1.5-1.8B** model from Alibaba. The training was performed on the IIIT Kottayam High-Performance Computing (HPC) cluster equipped with NVIDIA H100 GPUs.

---

## Objectives

1. Create a clean Tamil–English parallel corpus from multiple sources.  
2. Preprocess and normalize bilingual text for consistent tokenization.  
3. Fine-tune Qwen1.5-1.8B using PEFT LoRA adapters for efficient parameter optimization.  
4. Optimize the fine-tuning process for the IIIT Kottayam HPC environment.  
5. Evaluate tokenization efficiency and model performance across languages.

---

## Dataset Preparation

### 1. Datasets Used

| Dataset | Description | Size | Notes |
|----------|-------------|------|-------|
| `nlpc_uom_en_ta` | Tamil–English parallel corpus from University of Moratuwa | 62,852 | Removed metadata and null entries |
| `en_ta_118k.csv` | Bilingual corpus scraped from news and Wikipedia sources | 118,714 | Cleaned and deduplicated |
| `tamil_wikipedia_clean.jsonl` | Tamil Wikipedia monolingual data | 47,000 | Tamil-only corpus |
| `open_orca_clean.jsonl` | English instruction-tuning dataset (subset of OpenOrca) | 200,000+ | English monolingual corpus |
| `culturax_tamil_clean.jsonl` | Tamil texts from the Culturax dataset | 22,000 | Enriched Tamil representation |

All datasets were converted to JSONL format for compatibility with the Hugging Face `datasets` library.

---

### 2. Data Cleaning Pipeline

```python
Initial shape: (118759, 3)
After dropping nulls: (118759, 2)
After removing duplicates: (118743, 2)
After filtering short entries: (118714, 2)
Saved cleaned corpus: data/processed/en_ta_118k_clean.jsonl
Final size: 118714 rows
````

**Cleaning Steps:**

* Removed empty or null entries.
* Normalized UTF-8 encoding for both Tamil and English text.
* Stripped whitespace, punctuation, and formatting noise.
* Filtered out sequences shorter than 5 characters.
* Retained Romanized Tamil entries for bilingual balance.
* Merged all datasets into a unified Hugging Face `DatasetDict`.

---

### 3. Preprocessing Scripts

| Script                     | Purpose                                                      |
| -------------------------- | ------------------------------------------------------------ |
| `data_cleaning.py`         | Cleans and merges raw bilingual datasets.                    |
| `tokenization_analysis.py` | Benchmarks tokenization efficiency across languages.         |
| `prepare_dataset_hf.py`    | Converts processed data into Hugging Face-compatible format. |

---

## Tokenizer Evaluation

Tokenization efficiency was compared across multiple models to evaluate Tamil script coverage.

| Model                 | English Tokens/Char | Tamil Tokens/Char | Remarks                            |
| --------------------- | ------------------- | ----------------- | ---------------------------------- |
| Mistral-7B-Instruct   | 0.15                | 1.45              | Poor Tamil handling                |
| OpenHermes-Mistral    | 0.17                | 1.31              | Better Tamil support               |
| **Qwen1.5-1.8B-Chat** | **0.17**            | **1.08**          | Best overall bilingual performance |

**Decision:**
Qwen1.5-1.8B-Chat was selected for fine-tuning due to its balanced multilingual tokenizer, Unicode coverage, and manageable model size (1.8B parameters).

---

## Environment Setup (HPC)

Environment configuration on the HPC cluster (Ubuntu 20.04, NVIDIA H100, CUDA 12.5):

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -U pip setuptools wheel

# Install required packages
pip install torch transformers peft bitsandbytes datasets accelerate sentencepiece

# Verify GPU
nvidia-smi
```

**Versions:**

* Python: 3.10.14 (manually built)
* PyTorch: 2.4.1
* Transformers: 4.46.3
* PEFT: latest (installed via Git)
* BitsAndBytes: 0.45.5

Python 3.10 was compiled manually since the HPC’s default environment provided Python 3.8, which was incompatible with the latest PEFT release.

---

## Model Fine-Tuning Pipeline

### Script: `train_qwen_bilingual_hpc.py`

**1. Dataset Loading**

```python
dataset = load_dataset("json", data_files="data/processed/en_ta_118k_clean.jsonl")
```

**2. Tokenization**

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
```

**3. Model Loading (BF16 Precision for H100)**

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**4. LoRA Configuration**

```python
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
```

**5. Training Arguments**

```python
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    save_steps=1000,
    logging_steps=100,
    report_to="none"
)
```

**6. Training Execution**

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)
trainer.train()
```

---

## Model Configuration Summary

| Parameter          | Value                            |
| ------------------ | -------------------------------- |
| Base Model         | Qwen1.5-1.8B-Chat                |
| Fine-Tuning Method | LoRA (PEFT)                      |
| LoRA Rank (r)      | 16                               |
| LoRA Alpha         | 32                               |
| Dropout            | 0.05                             |
| Precision          | BF16                             |
| Optimizer          | AdamW                            |
| Learning Rate      | 2e-4                             |
| Scheduler          | Cosine Learning Rate             |
| Hardware           | NVIDIA H100 GPU                  |
| Framework          | Hugging Face Transformers + PEFT |

---

## Issues Encountered and Resolutions

| Issue                                              | Cause                              | Resolution                                |
| -------------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| Missing SentencePiece dependency                   | Tokenizer requirement              | Installed `sentencepiece`                 |
| PEFT attribute error (`memory_efficient_backward`) | Version mismatch                   | Upgraded PEFT via GitHub                  |
| CUDA allocator crash (`CUDACachingAllocator.cpp`)  | BitsAndBytes with H100             | Removed quantization, used BF16 precision |
| Python 3.8 incompatibility                         | PEFT ≥ 0.17 requires Python ≥ 3.10 | Built Python 3.10.14 from source          |

---

## Current Status

* Data collection and preprocessing: **Completed**
* Tokenizer analysis and model selection: **Completed**
* Environment setup on HPC: **Completed**
* LoRA fine-tuning in BF16 mode: **Operational**
* Quantized training: **Deprecated due to CUDA allocator issues**

Next step: Evaluate the fine-tuned model on bilingual benchmarks (FLORES-200 Tamil subset, translation and instruction-following tasks).

---

## Future Work

1. Evaluate model translation quality using BLEU and chrF++ scores.
2. Add supervised fine-tuning (SFT) on bilingual instruction data.
3. Explore quantization-aware export (4-bit GGUF) for inference.
4. Deploy inference pipeline on a lightweight environment.
5. Compare performance against Mistral and Gemma bilingual baselines.

---

## License

This project is licensed under the **Apache 2.0 License**.
Datasets are used in accordance with their respective licenses.

---



```

