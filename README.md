![accuracy_LLMs_talemaader-dataset](https://github.com/user-attachments/assets/be4f9ef9-2287-4797-8d4e-47495fc6aaea)


tl;dr: The GPT model series show increasing Danish language competency on Danish expressions and figures of speech, and the GPT-4o predicts the correct definition with an impressive 93% accuracy on the *talemaader*-dataset, made available by DSL. The Claude Sonnet 3 model, on the other hand, does not well perform on Danish with 69% accuracy, which is comparable to the older GPT-3.5-turbo model's 67%. Meta's Llama and Google's Gemini models sit in the mid-range with ~87% and ~80% accuracy, respectively.
### Evaluating LLMs on Danish Expressions with DSL's *talemaader*-dataset
Digitaliseringsstyrelsen and Det Danske Sprog- og Litteraturselskab (DSL) have developed the 'talemaader' dataset, which consists of 1000 Danish idioms and fixed expressions. Each expression comes with one true definition and three false definitions. This project aims to evaluate how well various Large Language Models (LLMs) can classify the correct definitions, thereby testing their ability to understand Danish idioms and fixed expressions.

This project evaluates the language competency of six different LLMs, including the larger GPT models, Gemini, Claude and Llama on DSL's 'talemaader - 1000 Danish expressions' dataset.

For an walkthrough of this project and its findings, please the notebook (here)[notebooks\walkthrough_project_findings.ipynb].

#### Dataset and license
The dataset is made available under a CC-BY license and can be downloaded via sprogteknologi.dk by [clicking here](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet). All credit goes to Det Danske Sprog- og Litteraturselskab (DSL) and Digitaliseringsstyrelsen.

This is my first project of this sort, so input, comments and suggestions are very welcome! 


# Guide: Running the Danish Idioms Evaluation Pipeline with New Models

This guide provides step-by-step instructions for evaluating a new LLM on the Danish idioms dataset, specifically focused on adding Claude 3.5 Sonnet to the evaluation.

## Prerequisites

- Python environment with required packages (see requirements.txt)
- API keys set as environment variables:
  - `ANTHROPIC_API_KEY` for Claude models
  - `OPENAI_API_KEY` for GPT models
  - `GOOGLE_API_KEY` for Gemini models
  - `LLAMA_API_KEY` for Llama models

## Step 1: Update Model Configurations

1. Open `src/config/model_configs.py` and add your new model configuration:

```python
MODEL_CONFIGS = {
    # ... existing models ...
    "claude-3-5-sonnet": {
        "model_name": "claude-3-5-sonnet-20241022",
        "max_tokens": 1,
        "temperature": 0
    }
}
```

## Step 2: Update Script Arguments

You need to update the allowed model choices in several scripts:

1. In `src/utils/run_model_predictions.py`, find the argparse section and add the new model:

```python
parser.add_argument('--model', type=str, default="gpt-4", 
                  choices=['gpt-4', 'gpt-4o', 'gemini', 'llama', 'claude', 'claude-3-5-sonnet'],
                  help='Model name to use for predictions')
```

2. Also in `src/utils/run_model_predictions.py`, update the model initialization:

```python
if model_name == 'gemini':
    self.model = GeminiModel()
elif model_name == 'llama':
    self.model = LlamaModel()
elif model_name in ['claude', 'claude-3-5-sonnet']:
    self.model = ClaudeModel(model_name="claude-3-5-sonnet-20241022" if model_name == 'claude-3-5-sonnet' else "claude-3-sonnet-20240229")
else:
    self.model = GPTModel(model_name=model_name)
```

3. In `src/utils/add_true_label.py`, update the model choices:

```python
parser.add_argument('--model', type=str, default="gpt-4", 
                  choices=['gpt-4', 'gpt-4o', 'gpt-4o-smaller-prompt', 'gemini', 'llama', 'claude', 
                           'gpt-3.5-one_shot', 'claude-3-5-sonnet'],
                  help='Model name to process (default: gpt-4)')
```

4. In `src/utils/process_discrepancies.py`, update the model choices the same way:

```python
parser.add_argument('--model', type=str, default="gpt-4", 
                  choices=['gpt-4', 'gpt-4o', 'gpt-4o-smaller-prompt', 'gemini', 'llama', 'claude', 
                           'gpt-3.5-one_shot', 'claude-3-5-sonnet'],
                  help='Model name to process (default: gpt-4)')
```

## Step 3: Run the Pipeline

Execute these commands in sequence:

1. Generate predictions (repeat until all 1000 expressions are processed):
```bash
python -m src.utils.run_model_predictions --model claude-3-5-sonnet --batch-size 100
```

2. Add true labels:
```bash
python -m src.utils.add_true_label --model claude-3-5-sonnet
```

3. Process discrepancies:
```bash
python -m src.utils.process_discrepancies --model claude-3-5-sonnet
```

4. Analyze misinterpretation types:
```bash
python results/predictions/misinterpretation_analysis.py
```

5. Generate misinterpretation overview:
```bash
python results/predictions/generate_misinterpretation_overview.py
```

6. Calculate accuracy metrics:
```bash
python results/metrics/calculate_accuracy.py
```

## Step 4: (Optional) Check Expression Order

If you want to ensure your predictions are in the same order as the original dataset:

```python
# Create compare_order.py with the script provided
python compare_order.py
```

## Output Files

The pipeline produces these output files:

1. Predictions: `data/predictions/predicted_labels_claude-3-5-sonnet.csv`
2. Gold standard comparisons: `data/predictions/predicted_and_gold_labels_claude-3-5-sonnet.csv`
3. Discrepancies analysis: `data/processed/only_discrepancies_claude-3-5-sonnet.csv`
4. Misinterpretation analysis: `results/predictions/misinterpretations_claude-3-5-sonnet.csv`
5. Final accuracy metrics: `results/metrics/model_accuracy.csv`

## Logs

Find detailed logs in:
- `logs/batch_claude-3-5-sonnet_*.log` (predictions)
- `logs/analysis_claude-3-5-sonnet_*.log` (true label analysis)
- `logs/discrepancies_claude-3-5-sonnet_*.log` (discrepancy processing)
