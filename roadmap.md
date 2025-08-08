
# Project Refactoring and Improvement Roadmap

This document outlines the steps taken to refactor and improve the Java to C# code translation project.

## 1. Initial Assessment and Refactoring

*   **Initial State:** The project was originally a single Jupyter Notebook.
*   **Goal:** Refactor the notebook into a structured, command-line-driven project.

### Actions Taken:

1.  **Project Restructuring:**
    *   Created a `src` directory to house the Python source code.
    *   Split the notebook's logic into three distinct scripts:
        *   `src/data_handler.py`: For loading and preparing the dataset.
        *   `src/train.py`: For model training and evaluation.
        *   `src/translate.py`: For performing translation with the trained model.

2.  **Modeling Correction:**
    *   Replaced the incorrect `RobertaForMaskedLM` with the proper `EncoderDecoderModel` for sequence-to-sequence tasks.
    *   Ensured the fine-tuned model is correctly saved after training and loaded for inference.

3.  **Evaluation Improvement:**
    *   Integrated BLEU score calculation into the training script for more robust model evaluation.

4.  **Documentation Update:**
    *   Updated the `README.md` to reflect the new project structure and provide clear instructions for training and translation.
    *   Removed the original, now obsolete, Jupyter Notebook.

## 2. Dependency and Environment Troubleshooting

*   **Goal:** Resolve execution errors and create a stable, reproducible environment.

### Actions Taken:

1.  **`ImportError` Resolution:**
    *   **Problem:** Encountered `ImportError: cannot import name 'load_metric' from 'datasets'`. This was due to the function being deprecated.
    *   **Solution:** Modified `src/train.py` to use the modern `evaluate` library instead. Updated `requirements.txt` to include `evaluate` and `sacrebleu`.

2.  **Dependency Isolation:**
    *   **Problem:** Dependency conflicts were occurring in the global `(base)` conda environment.
    *   **Solution:** Recommended and guided the creation of a dedicated conda virtual environment (`code_translation_env`) to isolate project dependencies.

3.  **NumPy Version Conflict:**
    *   **Problem:** The newly released NumPy 2.x caused a compatibility crash with the installed PyTorch version.
    *   **Solution:** Pinned the NumPy version in `requirements.txt` to `numpy<2` to ensure a compatible version is used.

4.  **Hugging Face Authentication Error:**
    *   **Problem:** An expired Hugging Face access token was causing a `401 Client Error`, preventing the download of the pre-trained model.
    *   **Solution:** Diagnosed the issue and provided clear instructions on how to generate a new token and log in via the `huggingface-cli login` command.

## 3. Memory and Training Debugging

*   **Goal:** Resolve memory-related training failures and diagnose why the model was not learning.

### Actions Taken:

1.  **Memory Exhaustion (`zsh: killed`):**
    *   **Problem:** The training process was being terminated by the OS due to excessive memory usage.
    *   **Solution:** Modified `src/train.py` to include several memory optimization techniques:
        *   Added command-line arguments for `batch_size`, `gradient_accumulation_steps`, `fp16`, and `max_length` to allow for fine-tuning resource usage.
        *   Set environment variables (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `OMP_NUM_THREADS`) within the script to manage memory on macOS.
        *   Added a `--gradient_checkpointing` flag to further reduce memory consumption.

2.  **Syntax Error Correction:**
    *   **Problem:** A syntax error (`logging_steps=.logging_steps`) was introduced during the memory optimization changes.
    *   **Solution:** Corrected the line in `src/train.py` to use the proper `args.logging_steps` variable.

3.  **Diagnosing Failed Training (Empty/Repetitive Output):**
    *   **Problem:** The model, despite completing the training run, produced empty or nonsensical repetitive output.
    *   **Solution:** Implemented a systematic debugging process:
        *   Created a dedicated `src/evaluate.py` script to separate evaluation from training.
        *   Fixed a bug in the evaluation script where the `attention_mask` was not being passed to the model, a common cause of repetitive output.
        *   When the issue persisted, updated the evaluation script to print raw token IDs, which revealed the model was only generating padding tokens.
        *   **Conclusion:** Determined that the training process itself was failing silently, resulting in an untrained, non-functional model.

4.  **Creating a Debugging Workflow:**
    *   **Problem:** Needed a way to quickly test if the training loop was functional without waiting for a full training run.
    *   **Solution:**
        *   Created a small `debug.java-cs.txt` dataset with only 10 examples.
        *   Modified `src/data_handler.py` to load this debug dataset.
        *   Added a `--debug` flag to `src/train.py` to easily switch to this small dataset, allowing for rapid overfitting tests.

5.  **Correcting Data Loading Logic:**
    *   **Problem:** The debug workflow failed with a `FileNotFoundError` because the data handler was not robust enough to handle cases where validation and test sets don't exist.
    *   **Solution:**
        *   Created a dedicated `data/debug` directory to hold the debug dataset, making the project structure cleaner.
        *   Updated `src/data_handler.py` to be more resilient. If validation or test files are not found in the specified `--data_dir`, it now defaults to using the training set for those purposes. This is ideal for the debug/overfitting case.
        *   Simplified `src/train.py` by removing the now-redundant `--debug` flag logic, as the data directory now controls the workflow.

## Current Status

The project's training and data handling scripts are now significantly more robust. The immediate next step is to run the training script, pointing it to the `data/debug` directory, to confirm that the model can successfully overfit on the small dataset. This will validate that the core training logic is sound before attempting a full training run again.

---

## Methodology: Training and Evaluation

This section details the technical workflow for training the translation model and evaluating its performance.

### 1. Core Model and Architecture

*   **Base Model:** The project uses `microsoft/codebert-base`, a pre-trained model optimized for understanding code.
*   **Architecture:** The model is configured as an `EncoderDecoderModel`. This is a standard sequence-to-sequence (seq2seq) architecture essential for translation tasks. The encoder processes the input Java code, and the decoder generates the output C# code.

### 2. Data Handling and Processing

*   **Data Source:** The dataset consists of parallel text files containing corresponding lines of Java and C# code.
    *   **Full Dataset:** `train.java-cs.txt.java`, `valid.java-cs.txt.java`, `test.java-cs.txt.java` (and their `.cs` counterparts) located in the project root.
    *   **Debug Dataset:** A 20-line subset of the training data located in the `data/debug/` directory.
*   **Loading:** The `src/data_handler.py` script reads these files. It is designed to be flexible:
    *   For a full run, it loads the `train`, `valid`, and `test` sets.
    *   For a debug run (when `--data_dir` is set to `data/debug`), it loads the `train` set and, because `valid` and `test` files are missing, it reuses the training data for validation and testing.
*   **Tokenization:** The `RobertaTokenizer` (from `codebert-base`) converts the raw code strings into numerical tokens that the model can understand. This includes padding and truncating sequences to a fixed `max_length` and creating an `attention_mask` to ensure the model ignores padding.

### 3. Training Process

*   **Script:** `src/train.py` is the main script for training.
*   **Framework:** It uses the `Seq2SeqTrainer` from the Hugging Face `transformers` library, which automates the training loop.
*   **Objective:** The model is trained to minimize the difference between its predicted C# code and the actual C# code from the training pairs. The loss function ignores padded tokens in the labels (where the label is -100).
*   **Key Parameters (configurable via command-line):
    *   `--model_name`: Specifies the base model to use (e.g., `microsoft/codebert-base`).
    *   `--data_dir`: The directory where the data files are located. This is the key to switching between a full and a debug run.
    *   `--output_dir`: The directory where the trained model and its checkpoints will be saved.
    *   `--percentage`: The fraction of the dataset to use for training and validation.
    *   `--batch_size`: The number of code pairs processed in each training step.
    *   `--num_epochs`: The number of times the model will see the entire training dataset.
    *   `--fp16`: Enables 16-bit floating-point precision to speed up training and reduce memory usage.

### 4. Evaluation and Testing

*   **Validation During Training:** The `sacrebleu` metric is used to calculate the BLEU score on the validation set at regular intervals during training. This helps monitor the model's progress and prevent overfitting.
*   **Final Testing (Post-Training):**
    *   **Script:** `src/run_evaluation.py` is used for testing a trained model.
    *   **Process:**
        1.  Loads the fine-tuned model and tokenizer from the specified `--output_dir`.
        2.  Takes a sample of Java code as input.
        3.  Uses the model's `.generate()` method to produce the C# translation.
        4.  The `attention_mask` is explicitly passed to the `generate` method to ensure correct handling of padded inputs.
        5.  The generated token IDs are decoded back into human-readable C# code and printed.
    *   **Goal:** To assess the model's performance on new, unseen code (ideally from the `test.java-cs.txt` set) to get a true measure of its translation capability.

## 4. Major Improvements and Model Switch

*   **Goal:** Address fundamental issues preventing the model from learning and improve overall stability.

### Actions Taken:

1.  **Model Architecture Switch:**
    *   **Problem:** Using an encoder-only model (`microsoft/codebert-base`) for an encoder-decoder task resulted in an uninitialized and untrainable decoder.
    *   **Solution:** Switched the default base model to `Salesforce/codet5-base`, which is a pre-trained encoder-decoder model specifically designed for code-to-code tasks. This ensures both encoder and decoder have meaningful initial weights.
    *   Updated `src/train.py` to use `AutoTokenizer` and `AutoModelForSeq2SeqLM` for flexible model loading.

2.  **Robust Special Token Handling:**
    *   **Problem:** Incorrect or missing `decoder_start_token_id` and `eos_token_id` could lead to repetitive or empty generation.
    *   **Solution:** Implemented more robust logic in `src/train.py` to dynamically set `decoder_start_token_id` and `eos_token_id` based on the tokenizer's available special tokens (`bos_token_id`, `cls_token_id`, `eos_token_id`, `sep_token_id`).

3.  **Evaluation Script Consistency:**
    *   **Problem:** The `src/run_evaluation.py` script was still using `RobertaTokenizer` and `EncoderDecoderModel`, which would be incompatible with the new `CodeT5` model.
    *   **Solution:** Updated `src/run_evaluation.py` to use `AutoTokenizer` and `AutoModelForSeq2SeqLM` for consistency with the training script and to ensure it can load the `CodeT5` model.

4.  **Generation Parameter Refinement:**
    *   **Problem:** The model was still producing repetitive output during evaluation, even after debug training.
    *   **Solution:** Added `no_repeat_ngram_size`, `length_penalty`, `num_return_sequences`, and `max_new_tokens` to the `model.generate()` call in `src/run_evaluation.py` to improve generation quality and prevent repetition.

5.  **Dependency Version Pinning:**
    *   **Problem:** Unpinned dependencies could lead to compatibility issues with future library updates.
    *   **Solution:** Pinned specific versions for `transformers`, `datasets`, `evaluate`, `sacrebleu`, and `torch` in `requirements.txt` to ensure a stable environment.

6.  **Data Alignment Check Utility:**
    *   **Problem:** Misaligned Java and C# code pairs could silently corrupt the training data.
    *   **Solution:** Created a new utility script `src/check_data_alignment.py` to verify that the input and output data files have the same number of lines, ensuring data integrity.

## Current Status

All major identified issues related to model architecture, training stability, and evaluation consistency have been addressed. The project is now configured to use a more appropriate pre-trained model (`Salesforce/codet5-base`) and includes robust data handling and evaluation practices. The next step is to re-run the debug training with these comprehensive changes to confirm that the model can now successfully learn and overfit on the small dataset. This will be the definitive test of our core training pipeline.
