# Problem Statement

## Objective:
In this task, you will transform an open-source language model (LLM) into an NER model. This exercise mimics a common scenario in industry where developers use generative models for discriminative tasks. Your goal will be to reframe the model to operate as a true NER model. This assignment is designed to evaluate your ability to build, fine-tune, and evaluate an ML model for a specific NLP task. While no specific implementation approach is required, your solution should demonstrate clear reasoning, technical skill, and good engineering practices.

## Steps:

1. **Select an NER Dataset**
   - Choose a publicly available dataset for NER.
   - Note: Please aim to keep the dataset manageable in size for the take-home context (preferably between 1,000 - 10,000 samples).

2. **Model Selection**
   - Choose an open-source transformer-based LLM (such as Llama3, Mistral etc).
   - Consider model size and inference time to balance between computational efficiency and performance.
   - Feel free to experiment across models.

3. **Transform the Model**
   - An off-the-shelf LLM will not suffice for our needs. Feel free to transform the model as you deem fit - including architecture, loss functions etc, in order to create a genuine NER model.
   - Note: This is not a prompt engineering exercise. While you are free to experiment with the input format, the focus is on transforming the model itself.

4. **Train the Model**
   - Train the model on the dataset you selected.
   - Feel free to leverage a parameter-efficient tuning method or full-finetuning methods, whichever you prefer.

5. **Evaluate the Model**
   - Provide a performance report of your model on the test set, including at least accuracy, precision, recall, and F1-score.
   - Provide an analysis of any common failure cases (e.g., misclassified samples).
   - Compare model performance to:
     - prompt engineering approach on ChatGPT or Anthropic. For this, you'll need to write a prompt.
     - The best in class approach/benchmark for your dataset. Usually, this will be a BERT model finetuned on your dataset.

6. **Document Your Approach**
   - Summarize the steps taken to adapt the LLM to this task, detailing any specific challenges encountered and how you addressed them.
   - Include a short discussion on:
     - Why you chose the dataset and model you did.
     - The primary differences in training the model versus as a regular language model.
     - Any limitations or possible improvements to your approach.

## Submission Requirements
- Code submission in a notebook format (Jupyter Notebook or similar).
- Documented and commented code to explain key transformations and decisions.
- A report summarizing the approach and results in the notebook or as a separate file.

## Evaluation Criteria
- Technical Soundness: Proper adaptation of the LLM to a the task.
- Model Performance: Reasonable performance on the chosen task, given dataset limitations.
- Documentation & Clarity: Clear explanation of steps, decisions, and any challenges.
- Innovation: Creative solutions to adapt the LLM as a classifier within the constraints of the task.

## Note
This task is designed to be completed in 4-6 hours. Focus on demonstrating a solid understanding of how to adapt generative models to discriminative tasks rather than hyper-optimizing results.