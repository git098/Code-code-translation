# Code-to-Code Translation: Java to C#

## Overview

This project implements a deep learning model for translating Java code snippets into C# code. It leverages the powerful Hugging Face Transformers library, specifically fine-tuning a pre-trained encoder-decoder model for this sequence-to-sequence task.

Through a rigorous debugging and optimization process, this project demonstrates a robust pipeline for code translation, addressing common challenges such as memory management, model initialization, and data handling.

## Key Features

*   **Java to C# Translation:** Translates Java code into functionally equivalent C# code.
*   **`Salesforce/codet5-base` Model:** Utilizes a state-of-the-art pre-trained encoder-decoder model specifically designed for code-related tasks, ensuring better initial performance and training stability compared to encoder-only models.
*   **Memory Optimization:** Incorporates techniques like mixed-precision training (`fp16`) and gradient checkpointing to manage memory usage during training, crucial for larger models and datasets.
*   **Robust Data Handling:** Includes a flexible data loading mechanism that can handle full datasets or small debug subsets, and automatically reuses training data for validation/testing if dedicated files are not present.
*   **Comprehensive Evaluation:** Integrates BLEU score calculation for quantitative evaluation of translation quality.
*   **Debugging Workflow:** Features a dedicated debug mode and utility scripts to quickly diagnose and resolve training and data-related issues.
*   **Version Stability:** Dependencies are pinned to specific versions to ensure reproducibility and prevent compatibility problems.

## Project Structure

```
. # Project Root
├── src/
│   ├── data_handler.py         # Handles dataset loading and preprocessing
│   ├── train.py                # Main script for model training
│   ├── run_evaluation.py       # Script for evaluating trained models
│   └── check_data_alignment.py # Utility to verify data file line counts
├── data/
│   └── debug/                  # Small subset of data for quick debugging
├── fine-tuned-model/           # Directory where the trained model is saved
├── train.java-cs.txt.java      # Java training data
├── train.java-cs.txt.cs        # C# training data
├── valid.java-cs.txt.java      # Java validation data
├── valid.java-cs.txt.cs        # C# validation data
├── test.java-cs.txt.java       # Java test data
├── test.java-cs.txt.cs         # C# test data
├── requirements.txt            # Python dependencies
├── roadmap.md                  # Detailed development roadmap and debugging log
├── README.md                   # Project overview and instructions
└── LICENSE                     # Project license
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```
    *(Remember to replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details.)*

2.  **Create a Conda Virtual Environment (Recommended):**

    ```bash
    conda create -n code_translation_env python=3.9
    conda activate code_translation_env
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets:**

    Ensure the `train.java-cs.txt.java`, `train.java-cs.txt.cs`, `valid.java-cs.txt.java`, `valid.java-cs.txt.cs`, `test.java-cs.txt.java`, and `test.java-cs.txt.cs` datasets are placed in the root directory of the project.

    *(Note: For debugging purposes, a small `data/debug` subset is also included.)*

## Usage

### 1. Check Data Alignment (Optional but Recommended)

Before training, you can verify that your Java and C# data files are correctly aligned (have the same number of lines):

```bash
python src/check_data_alignment.py
```

### 2. Train the Model

To train the model on the full dataset:

```bash
python src/train.py \
  --data_dir . \
  --output_dir ./fine-tuned-model \
  --num_epochs 32 \
  --batch_size 2 \
  --eval_steps 500 \
  --learning_rate 5e-5 \
  --gradient_checkpointing
```

*   **`--data_dir .`**: Specifies that the main datasets are in the current directory.
*   **`--output_dir ./fine-tuned-model`**: Where the trained model checkpoints will be saved.
*   **`--num_epochs`**: Number of training epochs.
*   **`--batch_size`**: Batch size for training and evaluation.
*   **`--eval_steps`**: Evaluate every N steps.
*   **`--learning_rate`**: Learning rate for the optimizer.
*   **`--gradient_checkpointing`**: Enables memory-saving gradient checkpointing.

*(Note: `--fp16` is not included in this command as it may cause issues on non-CUDA devices like macOS MPS. If you have a CUDA GPU, you can add `--fp16` for faster training.)*

### 3. Evaluate the Trained Model

To evaluate the fine-tuned model on an example Java code snippet:

```bash
python src/run_evaluation.py
```

This script is configured to load the model from `./fine-tuned-model` and translate a hardcoded Java example. You can modify `src/run_evaluation.py` to test different inputs or integrate it with your test dataset for a comprehensive evaluation.

### 4. Debugging Training Issues (Advanced)

If you encounter issues during full training, you can use the debug mode to quickly test if the core training loop is functional by overfitting on a tiny dataset:

```bash
python src/train.py \
  --data_dir ./data/debug \
  --output_dir ./debug-model \
  --num_epochs 10 \
  --batch_size 1 \
  --eval_steps 10 \
  --learning_rate 1e-3
```

This will train a model on the small `data/debug` dataset. You can then evaluate it using `python src/run_evaluation.py` (after temporarily changing `model_path` in `run_evaluation.py` to `./debug-model`).

## Requirements

See `requirements.txt` for a full list of pinned dependencies.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
