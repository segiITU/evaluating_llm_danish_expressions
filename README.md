### Evaluating LLMs on Danish Expressions with DSL's *talemaader*-dataset
Digitaliseringsstyrelsen and Det Danske Sprog- og Litteraturselskab (DSL) have developed the 'talemaader' dataset, which consists of 1000 Danish idioms and fixed expressions. Each expression comes with one true definition and three false definitions. This project aims to evaluate how well various Large Language Models (LLMs) can classify the correct definitions, thereby testing their ability to understand Danish idioms and fixed expressions.

#### Project Overview
This repository is focused on evaluating some of the major LLMs currently available in the market. The evaluation involves using the 'talemaader' dataset to assess each model’s accuracy in identifying the correct definitions among multiple choices.

#### Results
TBD

Models evaluated:
- gpt-4 (by OpenAI)
- gpt-4o (by OpenAI)
- gpt-o1-preview (by OpenAI)
- Claude Sonnet (by Anthropic)
- Gemini (by Google)

#### Current Status
The project is ongoing. Initial evaluations are being conducted, and detailed results will be documented in subsequent updates.
=======
This project evaluates the language competency of six different LLMs, including the larger GPT models, Gemini, Claude and Llama on DSL's 'talemaader - 1000 Danish expressions' dataset.

For a exploratory data analysis of the dataset, see please the EDA notebook in /notebooks.

#### Dataset and license
The dataset is made available under a CC-BY license and can be downloaded via sprogteknologi.dk by [clicking here](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet). All credit goes to Det Danske Sprog- og Litteraturselskab (DSL) and Digitaliseringsstyrelsen.
