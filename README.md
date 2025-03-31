# Danish Idioms Benchmark: Yes/No Prompt Variation

This branch contains a variation of the Danish Idioms benchmark that uses a Yes/No question format instead of multiple-choice questions. This allows us to study how the format of the prompt affects model performance on the same task.

## The Yes/No Approach

Instead of asking models to select the correct definition from four options (A, B, C, D), this variation asks models directly if a given definition matches the idiom. The prompt format is:

```
Betyder udtrykket {idiom} følgende: {definition}? (ja/nej)
```

For each idiom, we present the model with each of the four possible definitions separately, asking if each one matches. The model's responses (ja/nej) determine which definition it believes is correct.

This approach:
1. Changes the task from classification to binary verification
2. Potentially simplifies the reasoning process by focusing on one definition at a time
3. Might better reflect how humans verify their understanding of unfamiliar idioms

## Key Differences from the Original Benchmark

- **Prompt Structure**: Binary yes/no questions instead of multiple-choice selection
- **Processing**: Each idiom requires four API calls (one for each definition) instead of one
- **Storage**: Results are stored in separate directories to avoid overwriting the original benchmark results
- **Analysis**: The same accuracy metrics apply, but the process of reaching the predictions is different

## Running the Evaluation

The commands remain the same, but the code paths have been updated to use the yes/no approach:

```bash
# Generate predictions
python -m src.utils.run_model_predictions --model <model_name> --batch-size <5>

# Add true labels
python -m src.utils.add_true_label --model <model_name>

# Process discrepancies
python -m src.utils.process_discrepancies --model <model_name>

# Calculate accuracy metrics
python results/metrics/calculate_accuracy.py
```

## Output Files

Results will be stored in separate directories to avoid conflicts with the original benchmark:

- Predictions: `data/predictions/yesno/predicted_labels_<model_name>.csv`
- Gold standard comparisons: `data/predictions/yesno/predicted_and_gold_labels_<model_name>.csv`
- Discrepancies analysis: `data/processed/yesno/only_discrepancies_<model_name>.csv`
- Final accuracy metrics: `results/metrics/yesno_model_accuracy.csv`

## Research Questions

This variation helps us investigate:

1. Does phrasing the task as binary verification improve model performance?
2. Are models more accurate when focusing on one definition at a time?
3. Do different models respond differently to this change in prompt structure?
4. Does the same hierarchy of model performance maintain across different prompt formats?

The results will provide insights into both model capabilities and prompt engineering strategies.