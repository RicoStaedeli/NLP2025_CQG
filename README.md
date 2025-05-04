# Critical Question Generation: A novel approach by Cédric & Rico

We aim to generate useful critical questions for a given argumentative text.

**Critical Questions** are the set of inquiries that should be asked in order to judge if an argument is acceptable or
fallacious. Therefore, these questions are designed to unmask the assumptions held by the premises of the argument and
attack its inference.

This project is a development for the following shared
task: [CQG shared task](https://hitz-zentroa.github.io/shared-task-critical-questions-generation/)

## Table of Contents

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Project](#running-the-project)
6. [Team Contributions](#team-contributions)
7. [Results & Evaluation](#results--evaluation)
8. [References](#references)

---

## Project Description
![Project Architecture](Doc/Assets/Project%20Architecture.jpg)

### 1. Data Preprocessing - [Notebook Data Preprocessing](1_Preprocessing.ipynb)
First step of the project is the creation of a valid dataset for training the model.
For this we use the dataset SocraticQ: [SocraticQ](https://github.com/NUS-IDS/eacl23_soqg/tree/main)
The dataset includes short intervention texts and corresponding human-authored questions
#### Sample from the processed dataset (`train.csv`)
| **input** | **target (Question)**                                                                 |
|-------|-------------------------------------------------------------------------------------|
| implication_consequences: I'm referring only to aesthetics. | Are they obligated to make clothes that are as beautiful as possible?               |
| reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you. | How old are the kids who are screaming in public?                                   |
| implication_consequences: I think you have to live somewhere to know how it works. | Who should have the power to decide where the money goes?                           |
| implication_consequences: Its more than just income. | Is a family really entitled to live in prime real estate just because they want to? |
| clarity: I don’t believe that borders are actively making us safer. | What moral principle backs this view?   

### 2. Baseline - [Notebook Baseline](2_Baseline_CQS_generation.ipynb)
We generate baseline critical questions with pretrained LLMs for the validation dataset. To generate the baseline questions we use:
- LLama 3.1 8B Instruct 
- Qwen2.5 7B Instruct

Questions are generated for the interventions of the following dataset: [validation.json](Data/Processed/validation.json)

### 3. Training - [Notebook Training](3_Training.ipynb)
To fine-tune a language model for critical question generation, we used a parameter-efficient fine-tuning approach based on [QLoRA](https://arxiv.org/abs/2305.14314), facilitated by the `unsloth` framework for fast and memory-efficient training. Below is a summary of our training process:

#### Model & Framework
- **Base Model**: `unsloth/Llama-3.1-8B-Instruct`, a lightweight version of LLaMA tailored for instruction-following tasks.
- **Frameworks Used**: 
  - `unsloth` for efficient QLoRA fine-tuning
  - `transformers` & `trl` (SFTTrainer) for training and evaluation pipeline
  - `datasets` for data loading and handling
  - `peft` for managing parameter-efficient fine-tuning layers

#### Objective
Train the model to generate critical questions from intervention-based texts, using a supervised fine-tuning (SFT) approach on a curated training dataset.

#### Training Setup
- **Training Strategy**: Supervised fine-tuning with QLoRA adapters (low-rank matrices for efficient backpropagation)
- **Batching & Logging**:
  - Tensorboard used for logging training metrics
  - EarlyStopping callback configured to prevent overfitting
- **Precision**: Automatic detection of `bfloat16` for faster training on compatible GPUs

#### Output
- **Model Save Path**: Full fine-tuned model saved in a separate directory on Google Drive
- **LoRA Adapter Save Path**: Adapter weights separately saved for later inference or merging
- **Logs**: All training logs and metrics are saved locally and can be visualized via Tensorboard

#### Colab Integration
- The training pipeline is designed to work on Google Colab, including mounting Google Drive, token-based GitHub authentication, and auto-saving outputs to Drive.

![Training Workflow](Doc/Assets/Training%20Workflow.jpg)

### 4. Evaluation - [Notebook Evaluation](2a_Baseline_Evaluation.ipynb)
We define evaluation metrics and generate scores for the baseline models and the fine-tuned model. We evaluate the model with the following metrices:
- Semantic Similarity --> similarity between generated question and argumentative input text
- BLEURT: [Here](https://github.com/google-research/bleurt)
- ChatGPT 4.0 -> 
- Qualitative Sample Analysis --> (Two experts judge the generated questions)

### 5. RAG System
We create a complete pipeline to retrieve relevant information from open source document stores like Arxiv. 
This retrieved documents are stored in a vector database and retrieved when needed during the generation of the critical 
questions.

---

## Project Structure

```
.
├── Data/
│   ├── Raw/                              # Raw dataset
│   └── Processed/                        # Processed dataset
├── Doc/                                  # Documentation
├── Evluation/                
│   ├── Results/                          # Files with CQs from generation
│   ├── Scored/                           # Result scores from evaluations
├── Logs/                                 # Logs from all Notebooks               
├── Models/                               # Model checkpoints and exports
├── Utils/                                # Util files
├── .gitignore
├── 1_Preprocessing.ipynb                 # Preprocess Dataset and generate Training Data
├── 2_Baseline_CQS_Generation.ipynb       # Generate CQs with Baseline Models 
├── 2a_Baseline_Evaluation.ipynb          # Evaluate genertaed CQs from Baseline Models
├── 3_Training.ipynb                      # Fine-tune LLM on Training dataset
├── 3a_Finetuned_CQS_Generation.ipynb     # Generate CQs with finetuned LLM
├── 3b_Finetuned_Evaluation.ipynb         # Evaluate generated CQs from finetuned LLM
├── 4_Evaluation_Analytics.ipynb          # Analyse different evaluation metrics from all evaluations
├── README.md
└── requirements.txt
```

---

## Setup Instructions
To reproduce the proposed solution please run follow this instruction.

### Clone Repository
```bash
git clone https://github.com/RicoStaedeli/NLP2025_CQG.git
cd NLP2025_CQG
```

### Create Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix or MacOS
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

Follow these notebooks in order:
1. `1_Preprocessing.ipynb` - Data preprocessing
2. `2_Baseline.ipynb` - Establishing a baseline model
3. `3_Training.ipynb` - Model training
4. `4_Evaluation.ipynb` - Evaluating model performance

---

## Team Contributions

| Name          | Contributions                                  |
|---------------|------------------------------------------------|
| Rico Städeli  | Data preprocessing, baseline generation, parameter-efficient model fine-tuning  |
| Cédric Bohni  | Evaluation of CQs                              |


---

## Results & Evaluation

- [Briefly summarize your evaluation metrics, improvements from baseline, and insights drawn from experiments.]
- All detailed results are documented in `metrics/firstResults.json`.

---

## References

[List here any relevant papers, sources, libraries, or resources used in your project.]

- Doe, J. (2024). *Great NLP Paper*. Conference Name.
- [Library Used](https://example-library.com)

---
