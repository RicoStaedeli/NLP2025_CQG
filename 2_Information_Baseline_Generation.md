# Baseline Generation
Here we define the input and output of the generation of baseline creations.

## Data
To generate the baseline we use the evaluation dataset from the [CQG shared task](https://hitz-zentroa.github.io/shared-task-critical-questions-generation/). 
This dataset consists of 189 contexts with already generated and evaluated critical questions. 

## Model 
To generate the baseline we use LLama 3.1 8B Instruct from meta. To ensure faster and GPU efficient generation we use 
the 4bit Quantized version from unsloth. 

### Paramters
```python
#################################
####  Bits and Bytes Config #####
#################################
torch_dtype = torch.float16
device_map = "cuda"
quantization_config = {"load_in_4bit": True}
low_cpu_mem_usage = True

#################################
####  Model Config          #####
#################################
max_new_tokens = 256
do_sample = True
temperature = 0.7
top_p = 0.9

```
## Generaration
The generation is done with a few-shot prompt. We generate four distinct questions for every given context of the evaluation dataset. 
Each question is generated for one of our question types. 
The used prompts are:
```` python
schemas = {
    "CauseToEffect": """'Cause to Effect' with the examples:
    How strong is the generalisation that if <eventA> then <eventB>?
    Are there other factors in this particular case that could have interfered with the event of‘<eventB>’?
    """,

    "ExpertOpinion": """'Expert Opinion' with the examples:
    Is <expertE> a genuine expert in <domainD>?
    Did <expertE> really assert that <eventA>? Is <expertE>’s pronouncement directly quoted? If not, is a reference to the original source given? Can it be checked?
    If <expertE>’s advice is not quoted, does it look like important information or qualifications may have been left out?
    Is what <expertE> said clear? Are there technical terms used that are not explained clearly?
    Is <eventA> relevant to domain <domainD>?
    Is <eventA> consistent with what other experts in <domainD> say?
    Is <eventA> consistent with known evidence in <domainD>?
    """,

    "Analogy": """'Analogy' with the examples:
    Are <C1> and <C2> similar in the respect cited?
    Is <eventA> true in <C1>?
    Are there differences between <C1> and <C2> that would tend to undermine the force of the similarity cited?
    Is there some other case that is also similar to <C1>, but in which <eventA> is false?
    """,

    "FearAppeal": """'Fear Appeal' with the examples:
    Is <eventB> bad? Why and to whom is it bad?
    Is <eventA> away to prevent <eventB>?
    Is it practically possible for <eventA> to happen?
    Are there other consequences from <eventA>?
    """
}

messages = [
{"role": "system", "content": "You are a system designed to generate critical questions for a given argumentative context."},
{"role": "user", "content": f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Generate one critical question addressing the given context following the schema:

### Schema:
{schema}

Your answer is just the question without anything else.

This is the given context to relate the question to:

### Context:
{input_text}

### Response:
"""}   
]
   
````

## Output
The output is e json file containing the generated question for the given input context. 
Structure of the output:
```json
{
  "CLINTON_199_2": {
    "input": "CLINTON: \"which may prove to be an intelligence benefit\nwe've got to do everything we can to vacuum up intelligence from Europe, from the Middle East\nThat means we've got to work more closely with our allies, and that's something that Donald has been very dismissive of\nWe're working with NATO, the longest military alliance in the history of the world, to really turn our attention to terrorism\nWe're working with our friends in the Middle East, many of which, as you know, are Muslim majority nations\nDonald has consistently insulted Muslims abroad, Muslims at home, when we need to be cooperating with Muslim nations and with the American Muslim community\nThey're on the front lines\nThey can provide information to us that we might not get anywhere else\nThey need to have close working cooperation with law enforcement in these communities, not be alienated and pushed away as some of Donald's rhetoric, unfortunately, has led to\"",
    "cqs": [
      {
        "schema": "CauseToEffect",
        "cq": "Are there other factors in this particular case that could have interfered with the effectiveness of increased intelligence gathering from European and Middle Eastern allies, given the historical context of strained relationships between the US and these regions?"
      },
      {
        "schema": "ExpertOpinion",
        "cq": "Is what CLINTON said about Donald's rhetoric being alienating Muslim nations and communities clear, particularly regarding the implications of such rhetoric on intelligence gathering and cooperation in the fight against terrorism?"
      },
      {
        "schema": "Analogy",
        "cq": "Is there some other case where an administration's rhetoric and policies towards a particular group led to increased cooperation from that group, despite the administration's initial public criticisms of the group?"
      },
      {
        "schema": "FearAppeal",
        "cq": "Is it practically possible for increasing cooperation with Muslim nations and the American Muslim community, as Clinton suggests, to significantly reduce the threat of terrorism, and if so, what evidence supports this claim?"
      }
    ]
  },
    ...
}

```
