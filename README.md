# NLP2025_CQG

We propose a system that is able to generate critical questions for a given text.

# Project Goals

We aim to generate useful critical questions for a given argumentative text.

**Critical Questions** are the set of inquiries that should be asked in order to judge if an argument is acceptable or
fallacious. Therefore, these questions are designed to unmask the assumptions held by the premises of the argument and
attack its inference.

This project is a development for the following shared
task: [CQG shared task](https://hitz-zentroa.github.io/shared-task-critical-questions-generation/)

# Project constraints

This project is done in the context of the course NLP.

- Time limit of 5 Weeks
- Computing ressources: No GPU on development computer

# Project focus

- Develop a pipeline to infere with different pretrained LLMs
- Evaluate performance for zero-shot prompting, few-shot prompting and CoT
  prompting [Prompt techniques](https://www.promptingguide.ai/techniques)
- Finetune one LLM with a dataset containing critical Questions and evaluate this performance
- (optional): build a pipeline to use external knowledge, scrape actual data from internet or use a RAG system

# Project architecture

The following structure is aimed to create during this project.

![image](Doc/Assets/Untitled%20Diagram-3.jpg)

## Datasets:

- Subset of the US2016 corpus: [AIFdb US2016](https://corpora.aifdb.org/US2016)
- Subset of the Moral Maze corpus: [AIFdb MM2012](https://corpora.aifdb.org/mm2012)
- Subset of the args.me corpus: [zenodo Args.me](https://zenodo.org/records/4139439)
- Subset of the QT30 corpus: [AIFdb QT30](https://corpora.aifdb.org/qt30)
