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
> [!WARNING]  
> Describe Project overview with image
> Tell which model and dataset were used.

### 1. Data Preprocessing - [Notebook Data Preprocessing](1_Preprocessing.ipynb)
First step of the project is the creation of a valid dataset for training the model.
For this phase we use the dataset SocraticQ: [SocraticQ](https://github.com/NUS-IDS/eacl23_soqg/tree/main)

### 2. Baseline - [Notebook Baseline](2_Baseline_CQS_generation.ipynb)
We generate baseline critical questions with pretrained LLMs for the validation dataset. To generate the baseline questions we use:
- LLama 3.1 8B Instruct 
- Qwen2.5 7B Instruct

Questions are generated for the interventions of the following dataset: [validation.json](Data/Raw/sharedTask/validation.json)

### 3. Training - [Notebook Training](3_Training.ipynb)
We fine-tune a pretrained LLM with the generated dataset. We use unsloth to fine-tune the LLM. 

### 4. Evaluation - [Notebook Evaluation](2a_Baseline_Evaluation.ipynb)
We define evaluation metrics and generate scores for the baseline models and the fine-tuned model. We evaluate the model with the following metrices:
- Semantic Similarity
- BLEURT: [Here](https://github.com/google-research/bleurt)
- ChatGPT 4.0
- Human evaluation

### 5. RAG System
We create a complete pipeline to retrieve relevant information from open source document stores like Arxiv. 
This retrieved documents are stored in a vector database and retrieved when needed during the generation of the critical 
questions.

---

## Project Structure

```
.
├── Data/
│   ├── Raw/                  # Raw dataset
│   └── Processed/            # Processed dataset
├── Training/                 
│   ├── Checkpoints/          # Checkpoints of models
│   ├── Logs/                 # Logs of training
├── Evluation/                
│   ├── Results/              # Evaluation metrics and results
├── Models/                   # Model checkpoints and exports
├── Utils/
├── .gitignore
├── 1_Preprocessing.ipynb
├── 2_Baseline.ipynb
├── 3_Training.ipynb
├── 4_Evaluation.ipynb
├── README.MD
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
| Rico Städeli  | Data preprocessing, baseline model evaluation. |
| Cédric Bohni  | Model training, hyperparameter tuning.         |


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