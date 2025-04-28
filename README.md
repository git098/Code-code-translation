# Java to C# Code Translation using CodeBERT

## Overview

This project demonstrates how to translate Java code to C# code using the CodeBERT model. It utilizes the Hugging Face Transformers library to load the CodeBERT model and tokenizer, and then fine-tunes the model for sequence-to-sequence translation.

## Functionality

The project performs the following steps:

1.  **Data Loading:** Loads Java and C# code pairs from text files (`train.java-cs.txt.java`, `train.java-cs.txt.cs`, `valid.java-cs.txt.java`, `valid.java-cs.txt.cs`, `test.java-cs.txt.java`, `test.java-cs.txt.cs`).
2.  **Data Preprocessing:**
    *   Reduces the size of the dataset for faster training.
    *   Converts the data into Hugging Face Datasets.
    *   Tokenizes the Java and C# code using the CodeBERT tokenizer.
3.  **Model Training:**
    *   Loads the CodeBERT model for masked language modeling.
    *   Uses a data collator to prepare batches for training.
    *   Trains the model using the Seq2SeqTrainer.
4.  **Code Translation:**
    *   Loads the CodeBERT model for causal language modeling (sequence generation).
    *   Defines a function to translate Java code to C# code using the trained model.
    *   Provides an example of translating Java code to C# code.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/git098/Code-code-translation.git
    cd Code-code-translation
    ```

2.  **Download the Java and C# code datasets:**

    *   Download the `train.java-cs.txt.java`, `train.java-cs.txt.cs`, `valid.java-cs.txt.java`, `valid.java-cs.txt.cs`, `test.java-cs.txt.java`, and `test.java-cs.txt.cs` datasets and place them in the same directory as the notebook.
3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Jupyter Notebook:**

    *   Open `Java to C# code translation.ipynb` using Jupyter Notebook or JupyterLab.
    *   Run the cells in the notebook to perform the code translation.
2.  **Configure CodeBERT:**
    * The notebook uses CodeBERT for code translation. Ensure CodeBERT is properly configured.
3.  **Translate Java code to C#:**
    * Modify the `example_java_code` variable in the notebook to specify the Java code you want to translate.
    * Run the remaining cells to translate the Java code to C#.

## Requirements

*   Python 3.6+
*   `pandas`
*   `torch`
*   `datasets`
*   `transformers`

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
