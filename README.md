# Customer Complaint Classification Using Prompt Engineering and Optimization

## Project Overview

This project compares different prompt engineering techniques for classifying customer complaints using the locally running Llama 3.2 language model through Ollama.

The application allows users to enter a customer complaint and receive a predicted category. It also provides an experimental dashboard to evaluate and compare different prompting techniques on the same labeled dataset.

## Objectives

- Build a customer complaint classification application.
- Compare zero-shot, few-shot, role-based, structured, and optimized-candidate prompts.
- Evaluate classification performance using accuracy and macro F1-score.
- Compare response time and invalid model outputs.
- Identify the best-performing prompting technique on the evaluation dataset.

## Technologies Used

- Python
- Streamlit
- Ollama
- Llama 3.2 (3B)
- Pandas
- Scikit-learn

## Complaint Categories

The application uses seven categories:

1. Billing
2. Technical Issue
3. Delivery
4. Upgrade
5. Cancellation
6. Complaint
7. Other

## Project Structure

```text
custom_comp/
├── experiment.py
├── complaints_test.csv
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Install Python

Install Python 3.10 or a compatible version and ensure Python is available in your terminal.

### 2. Install Ollama

Download and install Ollama from:

https://ollama.com/download

### 3. Download the language model

Open your terminal and run:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before starting the application.

### 4. Install Python dependencies

Navigate to the project directory:

```bash
cd path/to/custom_comp
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 5. Run the application

Run the following command:

```bash
python -m streamlit run experiment.py
```

The application will open in your browser.

## Prompting Techniques Compared

- **Zero-shot prompting:** Classifies complaints without examples.
- **Few-shot prompting:** Includes labeled examples in the prompt.
- **Role-based prompting:** Assigns the model a customer-support classification role.
- **Structured prompting:** Provides explicit rules and output-format instructions.
- **Optimized-candidate prompting:** Tests a refined prompt designed to improve classification consistency.

The optimized candidate is not automatically the best prompt. Its performance must be measured against the other techniques.

## Evaluation Metrics

- **Accuracy:** The proportion of complaints classified correctly.
- **Macro F1-score:** The average F1-score across categories, giving each category equal weight.
- **Average response time:** The average time taken to classify a complaint.
- **Invalid outputs:** Responses that do not match the expected category labels.
- **Confusion matrix:** Shows correct and incorrect classifications for each category.

## Dataset

The `complaints_test.csv` file contains two required columns:

- `complaint`: The customer complaint text.
- `true_category`: The correct category used to evaluate the prediction.

The included starter dataset contains example complaints. For stronger academic results, use a larger, representative labeled dataset and keep a separate held-out test set for final evaluation.

## Experimental Methodology

1. Use the same labeled complaints for every prompting technique.
2. Use the same language model and generation settings.
3. Record each technique's predictions and response times.
4. Calculate accuracy, macro F1-score, and invalid-output counts.
5. Tune prompts on development data only.
6. Evaluate the selected prompt on a held-out test set.
7. Report the measured results without assuming that any technique will perform best in advance.

## Limitations

- Model predictions may be incorrect or inconsistent.
- Performance depends on dataset quality and category definitions.
- A small dataset may not represent real customer complaints.
- Local inference speed depends on the computer's hardware.
- Prompt optimization does not guarantee better results.

## Author

Data Science Student

## License

This project is intended for educational and research purposes.
