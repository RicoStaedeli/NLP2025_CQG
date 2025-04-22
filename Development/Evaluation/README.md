# Evaluation of Critical Questions
In this directory we have the code to evaluate the generated critical questions of our system.

## Run evaluation
To evaluate the generated questions please run the following script in your **root** directory of the project:
```sh
#!/bin/bash

python Evaluation/Evaluation/evaluation.py \
    --metric similarity \
    --input_path Datasets/sharedTask/sample.json \
    --submission_path Evaluation/Results/output_llama8.json \
    --threshold 0.6
```
