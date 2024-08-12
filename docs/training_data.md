## Accessing training data from LLMonPy

RankOutputStep is LLMonPypeline that is used to rank multiple outputs.  It uses a series of one-on-one comparisons to rank the outputs,
with the winner being the output with the most one-on-one victories.  Each contest produces output: question - best answer -
worse answer, that may be used to train the models.  [Research has shown that synthetic data derived from this process
can you used to improve the quality of future responses.](https://arxiv.org/html/2408.02666v1?utm_source=substack&utm_medium=email) 


LLMonPy training data is stored in the SQLite db, indexed by the name of the prompt that generated the outputs.  To get 
a list of prompts with training data, use the following command:

```bash
llmonpy qbawa_list
```

This command will return a list of prompt names, something like:

```
Step names with QBaWa data:
NameIterativeRefinementTournamentPrompt
```

To get the actual training data as JSON, use the following command:

```bash
llmonpy qbawa -name=NameIterativeRefinementTournamentPrompt
```

The qbawa command will return JSON an array of TourneyResult objects like in this file: [sample_training_data.json](sample_training_data.json)

### Structure of TourneyResult Data

```json
        "tourney_result_id": "76c88580-90dc-44b7-8d1c-0a5a3dadb7ec",
        "trace_id": "d3219870-e1bf-463c-9a47-72ca35ade23f",

        ...
        "input_data": {
            "judgement_prompt": "\n            I need you to judge the name suggestion for a new prompting technique.  The name should be descriptive, \n            but punchy and positive.  The name should not sound too technical or boring. The name should be easy to \n            remember.  The name should be easy to spell.  The name should be easy to say. Examples of good comparisons are:\n            \n            Artificial Intelligence vs. PatternSolver:  Artificial Intelligence is the winner\n            GenOpt vs. Genetic Algorithms:  Genetic Algorithms is the winner\n            Quicksort vs. EffiSort:  Quicksort is the winner\n            OnePassDetect vs. YOLO:  YOLO is the winner\n            PageRank vs TopPage: PageRank is the winner\n            SurrealVis vs Deep Dream: Deep Dream is the winner\n\n            Given these instructions, which do you think is the better name:\n            \n            Candidate 1: {{ contestant_1_name }} vs Candidate 2: {{ contestant_2_name }}\n            \n            Please reply with this JSON if Candidate 1 is the winner : {\"winner\": 1} or with this JSON if\n            Candidate 2 is the winner: {\"winner\": 2}.  Do not include any other text in your response.\n            ",
            "judgement_model_info_list": [
                {
                    "model_name": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                    "client_settings_dict": {
                        "temp": 0.0
                    }
                },
                {
                    "model_name": "gemini-1.5-flash",
                    "client_settings_dict": {
                        "temp": 0.0
                    }
                },
               ...
            ]
        },
```

Interesting fields, I've left out some data for brevity:

- **tourney_result_id**:  A unique identifier for the tourney result
- **trace_id**:  A unique identifier for the trace that generated the tourney
- **input_data**:  The data that was used to generate the tourney
  - **judgement_prompt**:  The prompt that was used to judge the outputs
  - **judgement_model_info_list**:  A list of models that were used to judge the outputs

```json
"contestant_list": [
            {
                "step_id": "0bf1a7ff-704e-4e84-a654-a8d820cd2d2f",
                "llm_model_info": {
                    "model_name": "gemini-1.5-flash",
                    "client_settings_dict": {
                        "temp": 0.75
                    }
                },
                "step_output": {
                    "name": "Adaptive Response Tournament (ART)"
                },
                "victory_count": 8
            },
            {
                "step_id": "70e064f2-878c-42da-a840-db11a36093eb",
                "output_id": "d737c781-c48f-4e61-80ce-e3f0292a207d",
                "llm_model_info": {
                    "model_name": "accounts/fireworks/models/mythomax-l2-13b",
                    "client_settings_dict": {
                        "temp": 0.75
                    }
                },
                "step_output": {
                    "name": "PromptTournament"
                },
                "victory_count": 7
            },
            ...
```

The contestant_list is an array of outputs that were judged in the tourney.  Each contestant has the following fields:
- **step_id**: The step_id within the trace that generated the output.  You can use this to get logs and other data about the output.
- **llm_model_info**:  The model and settings that generated the output
- **step_output**:  The response that is being ranked
- **victory_count**:  The number of one-on-one victories the output had in this tourney

```json
"contest_result_list": [
            ...
            {
                "step_id": "e42130f5-dbd9-40ca-a9b2-caea0af4961e",
                "contestant_one_output_id": "00b02bb5-a57f-4214-8ad6-2644ad1afdb6",
                "contestant_two_output_id": "2b38b7d0-5d4a-4454-b691-071f440c2ae9",
                "winner_output_id": "2b38b7d0-5d4a-4454-b691-071f440c2ae9",
                "dissenting_judges": 2
            },
```

The contest_result_list is an array of one-on-one contests that were used to rank the outputs.  
Each contest has the following fields:
- **step_id**: The step_id within the trace that generated the contest.  This id could be used to link to human annotation of
this contest.
- **contestant_one_output_id**:  The id of the first output in the contest
- **contestant_two_output_id**:  The id of the second output in the contest
- **winner_output_id**:  The id of the output that won the contest, this value will be either contestant_one_output_id or contestant_two_output_id
- **dissenting_judges**:  The number of judges that disagreed with the winner.  This is useful for identifying contests that may need human annotation.  Or you may not want to include contests with dissent in your training data.


### Reading the Training Data File
There is a helper function that makes is very easy to read the training data file.  Here is an example of how to use it:

```python
from llmonpy.trace_log import TourneyResult

if __name__ == "__main__":
    result_list = TourneyResult.from_file("sample_training_data.json")
    print("Results:")
    for result in result_list:
        print(result.to_json())
```

This code will read the training data file and convert the data to a list of TourneyResult objects.  This example then
iterates through the list and prints the JSON representation of each object.