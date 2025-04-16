# UNICODE
This repository provides the official implementation of the paper 'Unified Defense: Defending Renaming-Based Adversarial Attacks via Code Normalization' (published in The 2025 ACM SIGSAC Conference on Computer and Communications Security). We organize the code as follows:

📂 Code Structure

CODA/, ALERT/, CodeTAE/, CodeDenoise/

Implementation of each baseline method, with separate modules for different models and tasks.

code/

Our abstraction and instantiation implementations.

🗃 Data & Model Weights

Due to the large volume of experimental data and model files:

Currently, we provide the dataset for the clone detection task corresponding to the CodeDenoise to support reproducibility.
Model weights are temporarily excluded due to file size constraints.
All datasets and pre-trained weights will be released via [cloud storage (e.g., Google Drive)] after the double-blind review process.

<img src="https://raw.githubusercontent.com/anonymous123ok/UNICODE/refs/heads/main/overview.jpg" alt="drawing" width="800">


Usage Instructions

To leverage our abstract framework and instantiate methods, follow these steps:

Configure Paths
Modify the following variables in the code:
```python
input_path = "your/input/path"  # Replace with your input directory
output_path = "your/output/path"  # Specify desired output location
```
API Key Setup
Replace the placeholder with your LLM provider's API key:
```python
api_key = "your_api_key_here"  # e.g.deepseek,gpt4-o-mini
```
Execution
Run the core pipeline with:
```shell
cd code;
python abstract.py
python normalization.py
```
