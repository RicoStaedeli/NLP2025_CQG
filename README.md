# EICQE: Expert Independent Critical Question Evaluation

In this project we established a pipeline able to evaluate expert independent how strong a critical question is. We
further trained several different LLMs with different training approaches and are able to generate meaningful critical
questions for a given argumentative input text.

> **Critical Questions** are the set of inquiries that should be asked in order to judge if an argument is acceptable or
> fallacious. Therefore, these questions are designed to unmask the assumptions held by the premises of the argument and
> attack its inference.

This project was developed for
the [CQG Shared Task](https://hitz-zentroa.github.io/shared-task-critical-questions-generation/).

---

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

#### 1.1 Combine & preprocess

First step of the project is the creation of a valid dataset for training the model.
For this we use the dataset [SocraticQ](https://github.com/NUS-IDS/eacl23_soqg/tree/main) as a foundation. The dataset
includes short intervention texts and corresponding human-authored questions. This dataset is originally split into 3
chunks and needs to be combined.

##### Sample from the raw SocratiQ datest(`train_chunk_1.csv`)

| **input**                                                                                              | **target (Question)**                                                 |
|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| implication_consequences: I'm referring only to aesthetics.                                            | Are they obligated to make clothes that are as beautiful as possible? |
| reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you. | How old are the kids who are screaming in public?                     |


<br>

#### 1. 2 Scoring

We generate a dataset containing only entries with critical questions. To generate this dataset we have defined an
evaluation pipeline described in [Evaluation](3_Evaluation.ipynb).
The result is a dataset with 3'346 qualitative context-question pairs.

##### Below an excerpt of this filtered [dataset](Data/Processed/CQ%20SFT%20Dataset.json).

```json
[
  {
    "id": 1,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "CauseToEffect"
  },
  {
    "id": 2,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "FearAppeal"
  }
  ...
]
```

<br>
<br>

#### 1.3 Preference Dataset for Reinforcement Learning -  [Notebook Data Preprocessing](1_a_Preprocessing.ipynb)

We generated a second dataset to train the model with direct preferences. This dataset needs two columns, one is the
chosen and one is the rejected answer (In our case question).
The generation was done with `gpt-3.5-turbo` and the defined scoring pipeline from the [Evaluation](3_Evaluation.ipynb)
notebook.

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

Further information about preprocessing is in this [information](1_0_Information_preprocessing.md).

<br>
<br>

### 2. Question Generation - [Notebook Question Generation](2_Question_Generation.ipynb)

We create a generation script to generate question for the validation dataset. This script is used to generate critical
questions for all our trained models as well as for the baseline model.

- **Baseline Model**: `LLama 3.1 8B Instruct`
- **Validation Dataset**: [validation.json](Data/Raw/sharedTask/validation.json)

Further information about question generation are in the
following [information](2_0_Information_Question_Generation.md).

<br>
<br>

### 3. Training

To fine-tune our basemodel we have three approaches. First we started with a supervised fine-tuning approach to foster
the basics of critical question generation.
Second we used this SFT trained model to train it with a reinforcement learning approach called Direct Preference
Optimization. And third we use Odds Ratio Preference Optimization which is as well a reinforcement learning approach.

<img src="Doc/Assets/Training%20Workflow.png" alt="DPO dataset generation" style="width:60%;"/>

<br>
<br>

#### 3.1 Supervised Fine-tuning [Notebook SFT Training](4_Training_1_SFT.ipynb)

> First step is to use supervised fine-tuning to train the model to generate schema aligning critical questions.

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
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
    - **16bit merged model**: Full fine-tuned model saved in
      Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT)
    - **LoRA Adapter**: Adapter weights separately saved on
      Huggingface [ricostaedeli/Meta-Llama-3.1-1B-Instruct_SFT-lora](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_SFT-lora)

<br>
<br>

#### 3.2 Direct Preference Optimization Training [Notebook DPO Training](4_Training_2_DPO.ipynb)

> Direct Preference Optimization is a reinforcement learning strategy. As described in the
> paper [Direct Preference Optimization:
Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290).

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Dataset**
    - [Preference Dataset](Data/Processed/CQ%20DPO%20Dataset.json)
- **Setup**
    - **Training Strategy**: DPO fine-tuning with QLoRA adapters
- **Precision**:  `bfloat16`
- **Output**
    - **16bit merged model**: Full fine-tuned model saved in
      Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO)
    - **LoRA Adapter**: Adapter weights separately saved on
      Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO-lora](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_DPO-lora)

<br>
<br>

#### 3.3 ORPO Training [Notebook ORPO Training](4_Training_4_ORPO.ipynb)

> Odds Ratio Preference Optimization is a reinforcement learning training performed with a preference dataset.
- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Dataset**
    - [Preference Dataset](Data/Processed/CQ%20DPO%20Dataset.json)
- **Setup**
    - **Training Strategy**: ORPO fine-tuning with QLoRA adapters
- **Precision**:  `float16`
- **Output**
    - **16bit merged model**: Full fine-tuned model saved in
      Huggingface [ricostaedeli/Meta-Llama-3.1-8B-Instruct_ORPO](https://huggingface.co/ricostaedeli/Meta-Llama-3.1-8B-Instruct_ORPO)

<br>
<br>

### 4. Evaluation - [Notebook Evaluation](3_Evaluation.ipynb)

This notebook evaluates generated questions based on Walton's argumentation schemes. It scores question quality using a
combination of rule-based and model-based metrics.

- **Frameworks Used**:
    - `spaCy`: NLP pipeline for linguistic features
    - `nltk`: Tokenization and preprocessing, framenet and wordnet

The evaluation is further enhanced with an [analytical notebook](5_Evaluation_Analytics.ipynb).

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

- For all notebooks you need to create an access token for GitHub. This access token has to be placed in Google Colab as
  a secret with the name `GITHUB`.
- To access the baseline model `meta/Llama-3.1-8B-Instruct` you have to register for Meta Llama 3.1 Model Family on
  Hugging Face. Then create an access token in Hugging Face and place it in Google colab as secret with the name `
  HF_TOKEN`.
- Some Training notebooks have a connection to Weights & Biases to log the trainings. For these notebooks you need an
  additional API key from a Weights & Biases account. Place this API key in Google colab as a secret with the name `
  wandb`.
- The notebook [1_b_Generate_DPO_Dataset.ipynb](1_b_Generate_DPO_Dataset.ipynb) needs also an API key from OpenAI. Place
  this key as a secret in Google colab with the name `OPENAI_API_KEY`.
- All notebooks are running in google colab, and therefore you do not need a `requirements.txt` to install.

---

## Running the Project

If you want to reproduce the complete project you can follow these notebooks in following order. It is also possible to
use a notebook separately. The prerequisites are outlined in each notebook. It is highly not recommended to run the
preprocessing notebooks again.

```
1. 1_a_Preprocessing.ipynb                - Data preprocessing, ~4 hours
2. 1_b_Generate_DPO_Dataset.ipynb.ipynb   - Generate preference dataset, ~3 hours 
3. 2_Question_Generation.ipynb            - Generate questions with baseline model, ~1 hour
4. 3_Evaluation.ipynb                     - Evaluation of baseline model, ~10 minutes
5. 4_Training_1_SFT.ipynb                 - SFT fine-tuning, ~1 hour
6. 3_Evaluation.ipynb                     - Evaluation of SFT model, ~10 minutes
7. 2_Question_Generation.ipynb            - Generate questions with SFT model,, ~1 hour
8. 4_Training_2_DPO.ipynb                 - DPO fine-tuning, ~1 hour
9. 3_Evaluation.ipynb                     - Evaluation of DPO model, ~10 minutes
10. 2_Question_Generation.ipynb           - Generate questions with DPO model,, ~1 hour
11. 4_Training_3_DPO_from_SFT.ipynb       - DPO from SFT fine-tuning, ~1 hour
12. 3_Evaluation.ipynb                    - Evaluation of DPO from SFT model, ~10 minutes
13. 2_Question_Generation.ipynb           - Generate questions with DPO from SFT model,, ~1 hour
14. 4_Training_4_ORPO.ipynb               - OPRO fine-tuning, ~1 hour
15. 3_Evaluation.ipynb                    - Evaluation of ORPO model, ~10 minutes
16. 2_Question_Generation.ipynb           - Generate questions with ORPO, ~1 hour
17. 4_Training_5_ORPO_from_SFT.ipynb      - ORPO from SFT fine-tuning, ~1 hour
18. 3_Evaluation.ipynb                    - Evaluation of ORPO from SFT model, ~10 minutes
19. 2_Question_Generation.ipynb           - Generate questions with ORPO from SFT model, ~1 hour
20. 5_Evaluation_Analytics.ipynb          - Analyze all selected evaluation files, ~3 minutes
```

---

## Team Contributions

| Name         | Contributions                                                   |
|--------------|-----------------------------------------------------------------|
| Rico Städeli | Data Preprocessing, Model Training, Question Generation         |
| Cédric Bohni | Data Preprocessing, Evaluation & Analytics, Question Generation |

---

## Results & Evaluation

We were able to beat the baseline by more than 30% with our SFT trained model.

| Model    | Total Entries | Critical Count | % Critical | In Context Count | % In Context | Critical & In Context Count | % Critical & In Context |
|----------|---------------|----------------|------------|------------------|--------------|-----------------------------|-------------------------|
| Baseline | 744           | 237            | 31.9%      | 472              | 63.4%        | 132                         | 17.7%                   |
| SFT      | 744           | **486**        | **65.3%**  | 488              | 65.6%        | **281**                     | **37.8%**               |
| DPO      | 744           | 185            | 24.9%      | 536              | 72.0%        | 135                         | 18.1%                   |
| DPO_SFT  | 744           | 229            | 30.8%      | 546              | 73.4%        | 154                         | 20.7%                   |
| ORPO     | 744           | 210            | 28.2%      | **565**          | **75.9%**    | 154                         | 20.7%                   |

---

## References

- Calvo Figueras, Blanca, and Rodrigo Agerri. “Critical Questions Generation: Motivation and Challenges.” Proceedings of
  the 28th Conference on Computational Natural Language Learning, Association for Computational Linguistics, 2024, pp.
  105–16. DOI.org (Crossref), https://doi.org/10.18653/v1/2024.conll-1.9.
- Ruiz-Dolz, Ramon, and John Lawrence. “Detecting Argumentative Fallacies in the Wild: Problems and Limitations of Large
  Language Models.” Proceedings of the 10th Workshop on Argument Mining, Association for Computational Linguistics,
  2023, pp. 1–10. DOI.org (Crossref), https://doi.org/10.18653/v1/2023.argmining-1.1.
- Sellam, Thibault, et al. “BLEURT: Learning Robust Metrics for Text Generation.” Proceedings of the 58th Annual Meeting
  of the Association for Computational Linguistics, Association for Computational Linguistics, 2020, pp. 7881–92.
  DOI.org (Crossref), https://doi.org/10.18653/v1/2020.acl-main.704.
- Rafailov, Rafael, et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. arXiv:
  2305.18290, arXiv, 29 July 2024. arXiv.org, https://doi.org/10.48550/arXiv.2305.18290.

---
