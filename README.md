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

![Project Architecture](Doc/Assets/Project%20Architecture.png)

### 1. Data Preprocessing - [Notebook Data Preprocessing](1_Preprocessing.ipynb)

First step of the project is the creation of a valid dataset for training the model.
For this we use the dataset SocraticQ: [SocraticQ](https://github.com/NUS-IDS/eacl23_soqg/tree/main)
The dataset includes short intervention texts and corresponding human-authored questions

#### Sample from the raw SocratiQ datest(`train_chunk_1.csv`)

| **input**                                                                                              | **target (Question)**                                                               |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| implication_consequences: I'm referring only to aesthetics.                                            | Are they obligated to make clothes that are as beautiful as possible?               |
| reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you. | How old are the kids who are screaming in public?                                   |

#### Scoring

#### Sample from the processed dataset (`categoriesed_filtered_train_data.json`)
This dataset is filtered according a defined treshold during the scoring process.
```json
[
  {
    "id" : 1,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "CauseToEffect"
  },
  {
    "id" : 2,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "FearAppeal"
  }...
]
```

### 2. Baseline - [Notebook Baseline](2_Baseline_generation.ipynb)

We generate baseline critical questions with pretrained LLMs for the validation dataset. To generate the baseline
questions we use:

- LLama 3.1 8B Instruct

Questions are generated for the interventions of the following dataset: [validation.json](Data/Processed/validation.json)

### 3. Training 

To fine-tune our basemodel we have two approaches. First we started with a supervised fine-tuning approach to foster the basics of critical question generation. 
Second we used this SFT trained model to train it with a reinforcement learning approach called Direct Preference Optimization. 

#### 3.1 Supervised Fine-tuning 

##### Model & Framework

- **Base Model**: `unsloth/Llama-3.1-8B-Instruct`, a lightweight version of LLaMA tailored for instruction-following
  tasks.
- **Frameworks Used**:
    - `unsloth` for efficient QLoRA fine-tuning
    - `transformers` & `trl` (SFTTrainer) for training and evaluation pipeline
    - `datasets` for data loading and handling
    - `peft` for managing parameter-efficient fine-tuning layers

##### Objective

Train the model to generate critical questions from intervention-based texts, using a supervised fine-tuning (SFT)
approach on a curated training dataset.

##### Training Setup

- **Training Strategy**: Supervised fine-tuning with QLoRA adapters (low-rank matrices for efficient backpropagation)
- **Batching & Logging**:
    - Tensorboard used for logging training metrics
    - EarlyStopping callback configured to prevent overfitting
- **Precision**: Automatic detection of `bfloat16` for faster training on compatible GPUs

##### Output

- **Model Save Path**: Full fine-tuned model saved in a separate directory on Google Drive
- **LoRA Adapter Save Path**: Adapter weights separately saved for later inference or merging
- **Logs**: All training logs and metrics are saved locally and can be visualized via Tensorboard

#### Direct Preference Optimization
Direct Preference Optimization is a reinforcement learning strategy. 

#### Colab Integration

- The training pipeline is designed to work on Google Colab, including mounting Google Drive, token-based GitHub
  authentication, and auto-saving outputs to Drive.

![Training Workflow](Doc/Assets/Training%20Workflow.jpg)

### 4. Evaluation - [Notebook Evaluation](2a_Baseline_Evaluation.ipynb)

We define evaluation metrics and generate scores for the baseline models and the fine-tuned model. We evaluate the model
with the following metrices:

- Semantic Similarity --> similarity between generated question and argumentative input text
- BLEURT: [Here](https://github.com/google-research/bleurt)
- ChatGPT 4.0 ->
- Qualitative Sample Analysis --> (Two experts judge the generated questions)

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
├── 4_RAG System.ipynb                    # RAG System
├── 4a_RAG_CQS_generation.ipynb           # Gneratate CQs with RAG system
├── 4b_RAG_Evaluation.ipynb               # Evaluate CQs generated with RAG system
├── 5_Evaluation_Analytics.ipynb          # Analyse all evaluations
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

| Name         | Contributions                                                                                                                                 |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Rico Städeli | Data preprocessing, baseline and fine-tuned questin generation, parameter-efficient model fine-tuning with SFT and DPO,                       |
| Cédric Bohni | Data preprocessing, Creating a scoring system based on linguistic schema types, create evaluation pipeline and analysis forcritical questions |

---

## Results & Evaluation

- [Briefly summarize your evaluation metrics, improvements from baseline, and insights drawn from experiments.]
- All detailed results are documented in `metrics/firstResults.json`.

---

## References

- Calvo Figueras, Blanca, and Rodrigo Agerri. “Critical Questions Generation: Motivation and Challenges.” Proceedings of the 28th Conference on Computational Natural Language Learning, Association for Computational Linguistics, 2024, pp. 105–16. DOI.org (Crossref), https://doi.org/10.18653/v1/2024.conll-1.9.
- Ruiz-Dolz, Ramon, and John Lawrence. “Detecting Argumentative Fallacies in the Wild: Problems and Limitations of Large Language Models.” Proceedings of the 10th Workshop on Argument Mining, Association for Computational Linguistics, 2023, pp. 1–10. DOI.org (Crossref), https://doi.org/10.18653/v1/2023.argmining-1.1.
- Sellam, Thibault, et al. “BLEURT: Learning Robust Metrics for Text Generation.” Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, 2020, pp. 7881–92. DOI.org (Crossref), https://doi.org/10.18653/v1/2020.acl-main.704.
- Rafailov, Rafael, et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. arXiv:2305.18290, arXiv, 29 July 2024. arXiv.org, https://doi.org/10.48550/arXiv.2305.18290.

---
