# Preprocessing
In this section we define what steps were done and what the outcome of the preprocessing is.

## Data
We use the SocratiQ Dataset as raw data input. The Dataset consists of 84'582 context <--> question pairs. 
The original dataset is splittet into 3 chunks of 28'194 pairs. The dataset has three columns as depicted in the example:

### Example Raw SocratiQ Entry
|   | **input**                                                                                              | **target**                                                               |
|---|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 0 | implication_consequences: I'm referring only to aesthetics.                                            | Are they obligated to make clothes that are as beautiful as possible?             |
| 1 | reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you. | How old are the kids who are screaming in public?                                 |
| 2 | implication_consequences: I think you have to live somewhere to know how it works.                     | Who should have the power to decide where the money goes?                         |
| 3 | implication_consequences: Its more than just income.                                                   | Is a family really entitled to live in prime real estate just because they want to? |
| 4 | clarity: I don’t believe that borders are actively making us safer.                                    | What moral principle backs this view?                                             

## Processed Dataset for SFT Training
For the SFT Training we want to have a qualitative dataset containing only relevant question. 
We defined a scoring pipeline to evaluate each question in this dataset. 
The pipeline has a threshold to classify if the score for a specific question type is enough. 
The output of this step is a json file:

````json
{
  "id_1": {
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "context_token_len": 123
    "is_critical": true,
    "cause to effect" : 2,
    "Analogy": 5,
    "Expert Opinion": 12,
    "Fear": 3
  }
}

````

This generated JSON is used to create two new datasets for SFT Fine-tuning. 
### 1. Filtered dataset with schema
This dataset is in the structure of the original SocratiQ dataset with three columns:
```json
[
  {
    "id" : 1,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "best_schema": "Bias"
  },
  {
    "id" : 2,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "best_schema": "Fear"
  }...
]
```


### 2. Complete SocraticQ dataset with schema 
This dataset contains all entries from the original dataset but with additional columns. The columns are used during fine-tuning to give the model a better understanding of the relevance of the question.

```json
[
  {
    "id" : 1
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "best_schema": "Bias"
  },
  {
    "id" : 2,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "best_schema": ""
  }...
]
```

If the question is marked as not critical the new column is empty.

### 3. Complete SocraticQ dataset with scores
This dataset contains the complete SocratiQ dataset augmented with the scores generated with the evaluation script.
```json
[
  {
    "id" : 1,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "scores":    {
        "cause to effect" : 2,
        "Analogy": 5,
        "Expert Opinion": 12,
        "Fear": 3
    }
  }...
]
```

