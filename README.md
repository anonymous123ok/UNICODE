# UNICODE

## Contents

This repository provides the official implementation of the paper 'Unified Defense: Defending Renaming-Based Adversarial Attacks via Code Normalization' (in The 2025 ACM SIGSAC Conference on Computer and Communications Security). 

## Description

Deep code models (DCMs) are increasingly deployed in security-critical applications. Yet, their vulnerability to adversarial perturbations – such as subtle identifier renaming – poses significant risks, as these changes can induce out-of-distribution inputs and cause insecure predictions. A key challenge lies in defending against such attacks without prior knowledge of adversarial patterns, as the space of possible perturbations is virtually infinite, and conventional rule-based defenses fail to generalize. To address this challenge, we primarily focus on defending renaming-based adversarial attacks, which have the most significant impact on DCMs’ security, and propose a novel two-stage defense framework named UniCode, which proactively normalizes adversarial inputs into uniformly in-distribution representations. Specifically, the first stage strips all identifiers into placeholders, eliminating adversarial influence while maintaining the code structure and functionality, and the second stage reconstructs semantically meaningful identifiers by leveraging contextual understanding from large language models, ensuring the comprehensive code semantics are preserved. By fine-tuning the code models on the normalized distribution, UniCode renders models inherently robust against diverse renaming attacks without requiring attack-specific adaptations. To evaluate the performance of our approach, we have conducted a comprehensive evaluation by comparing it with state-of-the-art baseline methods. The experimental results demonstrate that UniCode achieves the best defense performance on 87.9% of subjects, with average improvements ranging from 17.3% to 126.0% over baselines in terms of defending effectiveness, indicating the superior performance of UniCode.

## Structure

Here, we briefly introduce the usage of each directory:

```
├─ALERT (Our baseline, for each baseline we provide code on three model/datasets due to space limited)
│  ├─codebert
│  ├─codet5
│  ├─graphcodebert
├─CDenoise (Our baseline, CodeDenoise)
├─CODA (Our baseline)
├─CodeTAE (Our baseline)
│─code (Our Approach, UniCode)
│  ├─abstract.py (abstracting identifier names)
│  ├─replace_method_name.py (abstracting function names)
│  ├─normalization.py (conducting code instantiation using LLM)
│  ├─build_txt.py (conducting data pre-processing)
│  ├─VulnerabilityPrediction_build_jsonl.py (conducting data pre-processing)
├─python_parser (parsing code for further analysis)
```

## Datasets/Models
Currently, we provide the dataset for the clone detection task to support reproducibility. Model weights are temporarily excluded due to file size constraints. All datasets and pre-trained weights will be released via cloud storage (e.g., Google Drive) after the double-blind review process. The overall model and datasets used in this paper has been listed below:

|           Task           |  Dateset   |   Train/Val/Test   |  acc   |
| :----------------------: | :--------: | :----------------: | :----: |
|     Clone Detection      |  CodeBERT  | 90,102/4,000/4,000 | 96.88% |
|                          |   GCBERT   |                    | 96.73% |
|                          |   CodeT5   |                    | 96.40% |
|                          | CodeT5Plus |                    | 97.47% |
| Vulnerability Prediction |  CodeBERT  | 21,854/2,732/2,732 | 63.76% |
|                          |   GCBERT   |                    | 64.13% |
|                          |   CodeT5   |                    | 59.99% |
|                          | CodeT5Plus |                    | 58.13% |
|    Defect Prediction     |  CodeBERT  |   27,058/–/6,764   | 84.37% |
|                          |   GCBERT   |                    | 84.89% |
|                          |   CodeT5   |                    | 88.82% |
|                          | CodeT5Plus |                    | 88.99% |

## Requirements:

- python==3.7.7
- tensorflow==2.3.0
- pytorch==1.5.1
- pandas==1.3.0

## Reproducibility

Usage Instructions

To leverage our abstract framework and instantiate methods, follow these steps:

Configure Paths Modify the following variables in the code:

```
input_path = "your/input/path"  # Replace with your input directory
output_path = "your/output/path"  # Specify desired output location
```

API Key Setup Replace the placeholder with your LLM provider's API key:

```
api_key = "your_api_key_here"  # e.g.deepseek,gpt4-o-mini
```

Execution Run the core pipeline with:

```
cd code;
python abstract.py --dataset <datasetname> --model <datasetname>
python normalization.py --dataset <datasetname> --model <datasetname>
```

`--dataset` refers to the dataset used for evaluation

`--model` refers to the model used for evaluation
