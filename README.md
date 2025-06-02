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

### 1. Data Preprocessing - [Notebook Data Preprocessing](1_a_Preprocessing.ipynb)

#### 1.1 Combine chunks and preprocess
First step of the project is the creation of a valid dataset for training the model.
For this we use the dataset SocraticQ: [SocraticQ](https://github.com/NUS-IDS/eacl23_soqg/tree/main)
The dataset includes short intervention texts and corresponding human-authored questions

##### Sample from the raw SocratiQ datest(`train_chunk_1.csv`)

| **input**                                                                                              | **target (Question)**                                                               |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| implication_consequences: I'm referring only to aesthetics.                                            | Are they obligated to make clothes that are as beautiful as possible?               |
| reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you. | How old are the kids who are screaming in public?                                   |


#### 1. 2 Scoring
We generate a dataset containing only entries with critical questions. To generate this dataset we have defined an evaluation pipeline described in [Evaluation](3_Evaluation.ipynb).
The result is a dataset with 3346 qualitative context-question pairs. 
##### Below an excerpt of this filtered [dataset](Data/Processed/CQ%20SFT%20Dataset.json).
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
#### 1.3 Preference Dataset for Reinforcement Learning -  [Notebook Data Preprocessing](1_a_Preprocessing.ipynb)
We generated a second dataset to train the model with direct preferences. This dataset needs two columns, one is the chosen and one is the rejected answer. In our case question.
The generation was done with `gpt-3.5-turbo` and the defined scoring pipeline. 

<img src="Doc/Assets/DPO%20dataset%20generation.png" alt="DPO dataset generation" style="width:35%;"/>

This resulted in a new dataset with 3'069 datapoints in the following structure:
```json

  {
    "id": 0,
    "prompt": [
      {
        "role": "user",
        "content": "Generate one critical question addressing the provided context. Ensure it matches the schema: Analogy\n\nContext: implication_consequences: The argument isn't that school teachers' compensation is adequate. The argument is that everyone should be paid a living wage. Someone asked how much a living wage was, and OP responded with an estimation of 40-50k."
      }
    ],
    "chosen": [
      {
        "role": "assistant",
        "content": "How much would a big Mac be if every employee made 50k a year?"
      }
    ],
    "rejected": [
      {
        "role": "assistant",
        "content": "Are school teachers' compensation and a living wage similar in terms of financial adequacy?"
      }
    ],
    "score_chosen": 4.5,
    "score_rejected": 6.5,
    "schema": "Analogy",
    "context": "implication_consequences: The argument isn't that school teachers' compensation is adequate. The argument is that everyone should be paid a living wage. Someone asked how much a living wage was, and OP responded with an estimation of 40-50k."
  }...
]
```

### 2. Baseline - [Notebook Question Generation](2_Question_Generation.ipynb)
We generate baseline critical questions with a pretrained LLM for the validation dataset. To generate the baseline
questions we use:

- LLama 3.1 8B Instruct

Questions are generated for the interventions of the following dataset: [validation.json](Data/Raw/sharedTask/validation.json)

### 3. Training 

To fine-tune our basemodel we have three approaches. First we started with a supervised fine-tuning approach to foster the basics of critical question generation. 
Second we used this SFT trained model to train it with a reinforcement learning approach called Direct Preference Optimization. And third we use Odds Ratio Preference Optimization which is as well a reinforcement learning approach.

#### 3.1 Supervised Fine-tuning [Notebook SFT Training](4_Training_1_SFT.ipynb)

- **Base Model**: `unsloth/Llama-3.1-8B-Instruct`, a lightweight version of LLaMA tailored for instruction-following
  tasks.
- **Frameworks Used**:
    - `unsloth` for efficient QLoRA fine-tuning
    - `transformers` & `trl` (SFTTrainer) for training and evaluation pipeline
    - `datasets` for data loading and handling
    - `peft` for managing parameter-efficient fine-tuning layers
- **Dataset**
  - [Filterd SocratiQ Dataset](Data/Processed/CQ SFT Dataset.json)
- **Setup**
  - **Training Strategy**: Supervised fine-tuning with QLoRA adapters
- **Precision**: `bfloat16` 
- **Output**
  - **16bit merged model**: Full fine-tuned model saved in Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT)
  - **LoRA Adapter**: Adapter weights separately saved on Huggingface [ricostaedeli/Meta-Llama-3.1-1B-Instruct_SFT-lora](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT-lora)


<img src="Doc/Assets/Training%20Workflow.jpg" alt="DPO dataset generation" style="width:60%;"/>


#### 3.2 Direct Preference Optimization Training [Notebook DPO Training](4_Training_2_DPO.ipynb)
Direct Preference Optimization is a reinforcement learning strategy. As described in the paper [Direct Preference Optimization:
Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290).

**Dataset** [DPO Dataset](Data/Processed/CQ%20DPO%20Dataset.json)
- **Setup**
  - **Training Strategy**: DPO fine-tuning with QLoRA adapters
- **Precision**:  `bfloat16` 
- **Output**
  - **16bit merged model**: Full fine-tuned model saved in Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO)
  - **LoRA Adapter**: Adapter weights separately saved on Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO-lora](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO-lora)

#### 3.3 ORPO Training [Notebook ORPO Training](4_Training_4_ORPO.ipynb)

**Dataset** [DPO Dataset](Data/Processed/CQ%20DPO%20Dataset.json)
- **Setup**
  - **Training Strategy**: ORPO fine-tuning with QLoRA adapters
- **Precision**:  `float16` 
- **Output**
  - **16bit merged model**: Full fine-tuned model saved in Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_ORPO](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_ORPO)




### 4. Evaluation - [Notebook Evaluation](2a_Baseline_Evaluation.ipynb)
This notebook evaluates generated questions based on Walton's argumentation schemes. It scores question quality using a combination of rule-based and model-based metrics.
- **Frameworks Used**:
  - `spaCy`: – NLP pipeline for linguistic features
  - `nltk`_ – Tokenization and preprocessing, framenet and wordnet
---

## Project Structure

- [Data/](./Data/)  
  &nbsp;&nbsp;&nbsp;&nbsp;- [Raw/](./Data/Raw/)  
  &nbsp;&nbsp;&nbsp;&nbsp;- [Processed/](./Data/Processed/)  
- [Doc/](./Doc/)  
- [Evluation/](./Evluation/)  
  &nbsp;&nbsp;&nbsp;&nbsp;- [Results/](./Evluation/Results/)  
  &nbsp;&nbsp;&nbsp;&nbsp;- [Scored/](./Evluation/Scored/)  
- [Logs/](./Logs/)  
- [.gitignore](./.gitignore)  
- [1_a_Preprocessing.ipynb](./1_a_Preprocessing.ipynb)  
- [1_b_Generate_DPO_Dataset.ipynb](./1_b_Generate_DPO_Dataset.ipynb)  
- [2_Question_Generation.ipynb](./2_Question_Generation.ipynb)  
- [3_Evaluation.ipynb](./3_Evaluation.ipynb)  
- [4_Training_1_SFT.ipynb](./4_Training_1_SFT.ipynb)  
- [4_Training_2_DPO.ipynb](./4_Training_2_DPO.ipynb)  
- [4_Training_3_DPO_from_SFT.ipynb](./4_Training_3_DPO_from_SFT.ipynb)  
- [4_Training_4_ORPO.ipynb](./4_Training_4_ORPO.ipynb)  
- [4_Training_5_ORPO_from_SFT.ipynb](./4_Training_5_ORPO_from_SFT.ipynb)  
- [5_Evaluation_Analytics.ipynb](./5_Evaluation_Analytics.ipynb)  
- [README.md](./README.md)   


---

## Setup Instructions
- For all notebooks you need to create an access token for GitHub. This access token has to be placed in Google Colab as a secret with the name "GITHUB".
- To access the baseline model `meta/Llama-3.1-8B-Instruct` you have to register for Meta Llama 3.1 Model Family on Hugging Face. Then create an access token in Hugging Face and place it in Google colab as secret with the name "HF_TOKEN"
- All notebooks are running in google colab and therefore you do not need a requirements.txt to install

---

## Running the Project
Follow these notebooks in order:
```
1. 1_a_Preprocessing.ipynb                - Data preprocessing, ~4 hours
2. 1_b_Generate_DPO_Dataset.ipynb.ipynb   - Generate preference dataset, ~3 hours 
3. 2_Question_Generation.ipynb            - Establishing a baseline model, ~1 hour
4. 3_Evaluation.ipynb                     - Evaluation of baseline model, ~10 minutes
5. 4_Training_1_SFT.ipynb                 - SFT fine-tuning, ~1 hour
6. 3_Evaluation.ipynb                     - Evaluation of SFT model, ~10 minutes
7. 4_Training_2_DPO.ipynb                 - DPO fine-tuning, ~1 hour
8. 3_Evaluation.ipynb                     - Evaluation of DPO model, ~10 minutes
9. 4_Training_3_DPO_from_SFT.ipynb        - DPO from SFT fine-tuning, ~1 hour
10. 3_Evaluation.ipynb                    - Evaluation of DPO from SFT model, ~10 minutes
11. 4_Training_4_ORPO.ipynb               - OPRO fine-tuning, ~1 hour
12. 3_Evaluation.ipynb                    - Evaluation of ORPO model, ~10 minutes
13. 4_Training_5_ORPO_from_SFT.ipynb      - ORPO from SFT fine-tuning, ~1 hour
14. 3_Evaluation.ipynb                    - Evaluation of ORPO from SFT model, ~10 minutes
15. 5_Evaluation_Analytics.ipynb          - Analyze all evaluation files, ~10 minutes
```


---

## Team Contributions

| Name         | Contributions                                                                                                                                  |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Rico Städeli | Data preprocessing, baseline and fine-tuned questin generation, parameter-efficient model fine-tuning with SFT and DPO,                        |
| Cédric Bohni | Data preprocessing, Creating a scoring system based on linguistic schema types, create evaluation pipeline and analysis for critical questions |

---

## Results & Evaluation

We were able to beat the baseline by more than 30% with our SFT trained model. 

| Metric              | Baseline | SFT     | DPO     |
|---------------------|----------|---------|---------|
| Scheme Accuracy     | 31.85%   | **65.32%** | 39.38%  |
| Context Accuracy    | 63.44%   | 65.59%  | **72.44%** |
| Scheme + Context    | 17.74%   | **37.77%** | 28.23%  |

---

## References

- Calvo Figueras, Blanca, and Rodrigo Agerri. “Critical Questions Generation: Motivation and Challenges.” Proceedings of the 28th Conference on Computational Natural Language Learning, Association for Computational Linguistics, 2024, pp. 105–16. DOI.org (Crossref), https://doi.org/10.18653/v1/2024.conll-1.9.
- Ruiz-Dolz, Ramon, and John Lawrence. “Detecting Argumentative Fallacies in the Wild: Problems and Limitations of Large Language Models.” Proceedings of the 10th Workshop on Argument Mining, Association for Computational Linguistics, 2023, pp. 1–10. DOI.org (Crossref), https://doi.org/10.18653/v1/2023.argmining-1.1.
- Sellam, Thibault, et al. “BLEURT: Learning Robust Metrics for Text Generation.” Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, 2020, pp. 7881–92. DOI.org (Crossref), https://doi.org/10.18653/v1/2020.acl-main.704.
- Rafailov, Rafael, et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. arXiv:2305.18290, arXiv, 29 July 2024. arXiv.org, https://doi.org/10.48550/arXiv.2305.18290.
---
