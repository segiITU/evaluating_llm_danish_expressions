tl;dr: The GPT model series show increasing Danish language competency on Danish expressions and figures of speech, and the GPT-4o predicts the correct definition with an impressive 93% accuracy on the *talemaader*-dataset, made available by DSL. The Claude Sonnet 3 model, on the other hand, does not well perform on Danish with 69% accuracy, which is comparable to the older GPT-3.5-turbo model's 67%. Meta's Llama and Google's Gemini models sit in the mid-range with ~87% and ~80% accuracy, respectively.
### Evaluating LLMs on Danish Expressions with DSL's *talemaader*-dataset
Digitaliseringsstyrelsen and Det Danske Sprog- og Litteraturselskab (DSL) have developed the 'talemaader' dataset, which consists of 1000 Danish idioms and fixed expressions. Each expression comes with one true definition and three false definitions. This project aims to evaluate how well various Large Language Models (LLMs) can classify the correct definitions, thereby testing their ability to understand Danish idioms and fixed expressions.

This project evaluates the language competency of six different LLMs, including the larger GPT models, Gemini, Claude and Llama on DSL's 'talemaader - 1000 Danish expressions' dataset.

For an walkthrough of this project and its findings, please the notebook (here)[notebooks\walkthrough_project_findings.ipynb].

#### Dataset and license
The dataset is made available under a CC-BY license and can be downloaded via sprogteknologi.dk by [clicking here](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet). All credit goes to Det Danske Sprog- og Litteraturselskab (DSL) and Digitaliseringsstyrelsen.

This is my first project of this sort, so input, comments and suggestions are very welcome! 
