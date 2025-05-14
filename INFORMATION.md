# Informationen zu Fine-Tuning

Ich habe mich über das fine tuning etwas eingelesen und folgende wichitge Punkte gufunden. Reinforcement learning wird in mehereren Schritten gemacht. 

## 1. Supervised Fine Tuning (SFT)

In diesem Schritt soll das model die Grundfunktion erlernen. 
Wir werden hier das Base Model Llama mit einfachen Instruction Tuning trainineren. Dazu verwenden wir das bestehende SocraticQ Dataset angereichert mit den Antworten- Typen.
Das Dataset sollte folgende Struktur haben:

Context | Question      | Type
-------- |---------------| --------
xyc   | What is that? | Cause to effect

Daraus wird dann ein Prompt erstellt für das Training:

````
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You will see a text and you should generate one critical question in the format {} for this text.

### Input:
{}

### Response:
{}"""
````

Mehr Informationen zu dem genutzten Trainer hier:
- [Huggingface SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Beispliel Llama 3.1 mit Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3)
- [Beispiel Llama 2](https://www.datacamp.com/tutorial/fine-tuning-llama-2)

## 2. Direct Preference Optimization Training (DPO)
Mit diesem Training soll das Model lernen, was eine gute und was eine schlechte Antwort ist. Dafür wird ein Dataset benötigt indem die Antworten (bei uns Fragen ) bewertet sind nach (chosen, rejected).
- chosen --> steht für eine gute antwort und chosen ist im Sinne einer menschlichen Präferenz gemeint. Ein Mensch würde diese Auswahl bevorzugen.

Das Dataset sol mit drei Spalten erstellt werden. 


Prompt | chosen        | rejected
-------- |---------------| --------
Erstelle eine Frage zu...   | What is that? | How are you?

Meine Überlegung ist es dies mit hilfe von ChatGPT zu erstellen. Es soll eine Pipeline erstellt werden die folgende Schritte ausführt:
1. Bewertung der bestehenden Frage aufgrund der definierten Evaluation (Stärke Cause to effect, etc.)
2. Frage hat einen hohen score für einen Typ
   1. Die Frage wird in die Spalte "chosen" geschrieben
   2. Der Prompt für das Datenset erstellt mit dem erkannten Typ. Bsp Prompt: Erstelle eine Frage mit dem Typ xy zu dem Kontext abc.
   3. Anschliessend wird über die ChatGPT API eine Frage erstellt die geneau nicht dem Typ entspricht 
   4. Mit unserer Evaluation wir geprprüft ob der Score wirklich nicht gut ist 
      1. Wenn Frage doch gut ist wiederhole ChatGPT request
      2. Wenn Frage schlecht die erstellete Frage in die spalte "rejected" schreiben


Mehr Informationen:
- [Huggingface Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats#preference)
- [Example DPO Training mit LLama 3](https://www.analyticsvidhya.com/blog/2024/05/fine-tune-llama-3-using-direct-preference-optimization/)


## 3. PPO Trianing mit reward function
I diesem Schritt wird eine Trainingsloop mit einer reward Funktion erstellt. Dazu wird folgendes benötigt:

- SocraticQ Dataset
- funktionierendes Scoring System um einer Frage während dem Training ein Score zuweisen zu können. 

Mehr Informationen
- [TRL PPO Trainer](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)

