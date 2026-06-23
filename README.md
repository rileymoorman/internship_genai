# patient_consult



## Background

This project was built during week two of my GenAI curriculum.The goal is to learn how AI agents work by building one that could give medical advice using a local llm and hardcoded tools.



### Function

The script uses a patient prompt to calculate bmi and lookup drug interactions, and returns a recommendation.



### calculate_bmi

uses the inputs height and weight converts the value into a decimal and calculates bmi. returns an error message if height or weight is less than zero. Uses if/else statement to categorize bmi. returns bmi, category, weight\_kg, height\_m.



### lookup_drug_interaction

cleans drug names- removes spaces, makes lowercase

tries both orders

searches through json files

finds a match and returns interactions, if not says "no known interaction found"



### TOOLS

the list that tells ollama what functions its allowed to call and how to use them. Each tool has a name, description, and what input it needs



TOOL_MAP:

connects tool names to python functions -agent knows what to run



run_agent:

inputs prompt and loops- sends prompt to llm, calling whichever tools it needs, repeats until llm has enough info to return a final summary

