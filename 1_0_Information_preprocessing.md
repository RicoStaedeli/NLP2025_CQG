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
[
    {
    "id":1,
    "context": "alternate_viewpoints_perspectives: A parallel argument would state that England is worse off because.",
    "question": "What about nations who have nothing?",
    "context_token_len":126,
    "is_Critical":false,
    "CauseToEffect":0.0,
    "Analogy":2.0,
    "ExpertOpinion":0.0,
    "FearAppeal":2
  }...
]

````
Found in Data/Final/scored/processed_train_data.json

This generated JSON is used to create two new datasets for SFT Fine-tuning. 
### 1. Filtered dataset with schema [CQ SFT Dataset](Data/Processed/CQ%20SFT%20Dataset.json)
This dataset is in the structure of the original SocratiQ dataset with three columns:
```json
[
  {
    "id" : 1,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "Bias"
  },
  {
    "id" : 2,
    "context": "reasons_evidence: If you are genuinely struggling and need help, someone is going to want to help you.",
    "question": "How old are the kids who are screaming in public? ",
    "schema": "Fear"
  }...
]
```


### 2. Complete SocraticQ dataset with scores - [CQ FULL Dataset](Data/Processed/CQ%20FULL%20Dataset.json)
This dataset contains the complete SocratiQ dataset augmented with the scores generated with the evaluation script.
```json
[
  {
    "id":1,
    "context":"alternate_viewpoints_perspectives: A parallel argument would state that England is worse off because America passed them as the dominant economic power in the world. Is the argument remotely true? Not at all - England has become far wealthier from trade with America. I'm typing this reply on a Samsung phone, made in Korea, which was an undeveloped country in the not too distant past. Who knows, maybe Vietnam has some kids now who will figure out a better smartphone. I'd love it if they became wealthy enough to actually make one. Your argument is essentially mercantilist, by thinking that trade is zero-sum, but actually trade improves income for all nations.",
    "question":"What about nations who have nothing?",
    "context_token_len":139,
    "is_Critical":false,
    "CauseToEffect":0.0,
    "Analogy":2.0,
    "ExpertOpinion":0.0,
    "FearAppeal":2
  }...
]
```

