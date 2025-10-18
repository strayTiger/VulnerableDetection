
# Vulnerability Detection

Vulnerability Detection is a project designed to compare the performance of different models in vulnerability detection tasks. The project implements and evaluates multiple vulnerability detection models, helping researchers and developers understand the advantages and limitations of these models.

## 🔑 Key Features

- **Multiple Model Comparison**: Implements various mainstream models (such as XLNet, CodeBERT, etc.) and compares their performance in vulnerability detection tasks.Based on the comparison results, the effectiveness of XLNet, which is based on Transformer-XL, is demonstrated, and an end-to-end framework XLNetVD is proposed.
- **Dataset Support**: Supports multiple vulnerability detection datasets, allowing users to test and validate models.
- **Efficient Performance Evaluation**: Uses standard evaluation metrics (such as F1-score) to assess model performance in vulnerability detection tasks.
- **Flexible Configuration and Extensibility**: The project supports flexible configuration, allowing users to customize and extend it according to their needs.

## ⚙️ Tech Stack

- **Python**: The programming language used to implement all functionality and models.
- **PyTorch**: A deep learning framework used for model training and inference.
- **Transformers**: A library for loading and applying pre-trained language models (such as CodeBERT and XLNet).
- **scikit-learn**: A machine learning library used for model evaluation and metric computation.
- **matplotlib**: A library for visualizing results.

## 📦 Pre-trained Weights

The pre-trained weights used for the models come from official model repositories, including:

- **RoBERTa**: [roberta-base](https://huggingface.co/roberta-base)
- **CodeBERT**: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **GraphCodeBERT**: [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base)
- **CodeT5**: [Salesforce/codet5-base](https://huggingface.co/Salesforce/codet5-base)
- **GPT2**: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- **XLNet**: [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)

All experiment code is written in **Python 3.11**, and model training and evaluation processes are facilitated using **PyTorch 2.9.0** and **Scikit-learn 1.7.0**.

## 📁 Directory Structure

```
Vulnerability-Detection/
├── data/        # Directory for datasets
├── model/       # Directory for model code
├── requirements.txt  # Dependency file
└── README.md    # Project description file
```
#### model/
It includes the code for all the models used for running and analysis.
- Model.py: Used to test the performance of various baseline models.
- loraModel.py: Used to test the performance of various models after LoRA fine-tuning.
- cweModel.py: Used for detecting different types of vulnerabilities.
- train.jsonl,valid.jsonl,test.jsonl: Partial real-world dataset.
- *_model: Folders where the models are stored. Please download them yourself.
#### data/
This folder contains a part of the SARD dataset.Used for detecting different types of vulnerabilities.
- train: Training set
- val: validation set
- test: test set

For the complete dataset, please request it via this email: **guanjun.lin@fjsmu.edu.cn**

## 🚀 Usage

### 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/strayTiger/Vulnerability-Detection.git
cd Vulnerability-Detection
```

### 2. Install Dependencies

Make sure you have a Python environment, and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Models

Follow the instructions in the `model/` directory to choose and run the appropriate model. Typically, you can use a command like the following to start training and evaluation:

```bash
python model/Model.py
```

For detecting different types of vulnerabilities, typically, you can use the following command to start training and evaluation:

```bash
python cweModel.py --data-root "E:\data" --model "*_model" --max-len 768 --batch-size 28 --epochs 3 --shuffle --strip-comments --strip-strings --mask-sard-hints
```

> Adjust the command according to the actual file path and configuration.

## 🛠️ Contributing

We welcome issues and pull requests! Please follow these steps to contribute:

1. Fork this repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## 📬 Contact

- Email: zhaoyinhu122@163.com
- GitHub: [strayTiger](https://github.com/strayTiger)

---

### 📜 License

This project's code is open-sourced under the [MIT License](./LICENSE). Anyone is free to use, copy, and modify it, as long as they include the original author(s) attribution.
