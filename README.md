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
> Tell which model and dataset was used.

### 1. Data Preprocessing
First step of the project is the creation of a valid dataset for training the model. 

### 2. Baseline
We generate baseline critical questions with pretrained LLMs for the validation dataset.

### 3. Training
We fine-tune a pretrained LLM with the generated dataset. 

### 4. Evaluation
We define evaluation metrics and generate scores for the baseline models and the fine-tuned model.

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

> [!NOTE]  
> This is only a Template. And you can add notes, Warnings and stuff with this format style. ([!NOTE], [!WARNING], [!IMPORTANT] )

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