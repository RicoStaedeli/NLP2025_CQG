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
    "CauseToEffect": """Examples are:
    What if the job is one that will cause their illness to manifest and affect their job performance?
    If so, do you think the lack of any of the pieces influence the decisions they make about whether to create the digital widget or which digital widget to create?
    If miners and the working class are such a small percentage of the population as to not make a difference in a straight up popular vote, then why do we let them have so much influence?
    """,

    "ExpertOpinion": """'Examples are:
    Do you have a scientific source that confirm that women only wants one partner to a greater degree than men?
    If you are just stating facts, the surely you can point to some studies that show most women are narcissistic sociopaths?
    Where in any scientific textbook or journal have you seen evidence of sex being described as a spectrum?
    """,

    "Analogy": """Examples are:
    If so, would these groups resemble what are commonly referred to as races?
    If whites are guilty of enjoying the benefits of stolen land after being here for two generations, why would that not mean a 2nd generation American POC not have a similar benefit?
    But why is the way her hair naturally grows from her head seen as less professional than chemically altering to mimic straight white European hair?
    """,

    "FearAppeal": """'Examples are:
    If he did such a great job making peace, why do we need sanctions to pressure North Korea into stopping its violent threats?
    And if we should strive to stop all killings then why would the implement of this murder be spared if it causes a very large number of deaths but was intended to cause none?
    If you say you can, then are you implying that you see Islam as being no worse?
    """
}


 "prompt": [
    {
        "role": "system",
        "content": "You generate concise, critical, single-sentence questions for argumentative contexts, matching specified question schemas."
    },
     {
        "role": "user",
        "content": f"Generate one critical question addressing the provided context. Ensure it matches the schema:{schema_key}{schema_text}.\n\nContext: {context} Respond only with the question and nothing else."
    }

],
   
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
