# Code Translation: Java to C#

This project fine-tunes Salesforce's **CodeT5-base** model for translating Java code snippets into C# code. CodeT5 is a pre-trained encoder-decoder transformer specifically designed for code understanding and generation tasks. The project leverages the Hugging Face Transformers library to fine-tune this model on Java-to-C# translation pairs.

## 🌟 Features

- **CodeT5 Fine-tuning**: Fine-tunes Salesforce's CodeT5-base model, a specialized transformer for code tasks
- **Memory Optimization**: Incorporates mixed-precision training (fp16) and gradient checkpointing for efficient memory usage
- **Robust Data Handling**: Flexible data loading with support for debug subsets and automatic fallback mechanisms
- **Comprehensive Evaluation**: Integrated BLEU score calculation for quantitative translation quality assessment
- **Debugging Workflow**: Dedicated debug mode for quick issue diagnosis and resolution
- **Version Stability**: Pinned dependencies ensure reproducibility and compatibility

## 📁 Project Structure

```
. # Project Root
├── src/
│   ├── data_handler.py         # Dataset loading and preprocessing
│   ├── train.py               # Main training script
│   ├── run_evaluation.py      # Model evaluation script
│   └── check_data_alignment.py # Data verification utility
├── data/
│   └── debug/                 # Small debug dataset
├── fine-tuned-model/          # Trained model output directory
├── train.java-cs.txt.java     # Java training data
├── train.java-cs.txt.cs       # C# training data
├── valid.java-cs.txt.java     # Java validation data
├── valid.java-cs.txt.cs       # C# validation data
├── test.java-cs.txt.java      # Java test data
├── test.java-cs.txt.cs        # C# test data
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT License
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/git098/Code-code-translation.git
   cd Code-code-translation
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   conda create -n code_translation_env python=3.9
   conda activate code_translation_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data alignment:**
   ```bash
   python src/check_data_alignment.py
   ```

## 🔧 Usage

### Training

Train the model on the full dataset:

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

#### Training Parameters

- `--data_dir .`: Location of training datasets
- `--output_dir ./fine-tuned-model`: Model checkpoint save directory
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training and evaluation batch size
- `--eval_steps`: Evaluation frequency (every N steps)
- `--learning_rate`: Optimizer learning rate
- `--gradient_checkpointing`: Enable memory-efficient gradient checkpointing
- `--fp16`: Enable mixed-precision training (CUDA GPUs only)

### Evaluation

Evaluate the trained model:

```bash
python src/run_evaluation.py
```

This script loads the model from `./fine-tuned-model` and translates a sample Java code snippet. Modify the script to test different inputs or integrate with your test dataset.

### Debug Mode

For quick testing and debugging:

```bash
python src/train.py \
    --data_dir ./data/debug \
    --output_dir ./debug-model \
    --num_epochs 10 \
    --batch_size 1 \
    --eval_steps 10 \
    --learning_rate 1e-3
```

This trains on a small debug dataset to verify the training pipeline functionality.

## 📊 Model Architecture

The project fine-tunes **Salesforce's CodeT5-base** (`Salesforce/codet5-base`), a specialized pre-trained encoder-decoder transformer model designed specifically for code understanding and generation tasks. 

### Why CodeT5?

CodeT5 offers several advantages for code translation:

- **Code-specific pre-training**: Unlike general-purpose language models, CodeT5 was pre-trained on massive code datasets with code-aware objectives
- **Identifier-aware**: The model understands programming language identifiers, keywords, and syntax patterns
- **Encoder-decoder architecture**: Perfect for sequence-to-sequence tasks like code translation
- **Multi-language support**: Pre-trained on multiple programming languages including Java and C#
- **Strong baseline performance**: Provides excellent starting weights for fine-tuning on specific translation tasks

### Fine-tuning Process

This project takes the pre-trained CodeT5-base model and fine-tunes it on parallel Java-C# code pairs to specialize it for Java-to-C# translation. The fine-tuning process adapts the model's weights to better understand the specific syntactic and semantic differences between Java and C#.

## 💾 Data Format

The training data consists of parallel Java and C# code files:
- `.java` files contain Java source code
- `.cs` files contain corresponding C# translations
- Each line represents a code snippet pair
- Files must have matching line counts

## 📈 Performance

The model evaluation includes:
- BLEU score calculation for translation quality
- Comprehensive testing on held-out test sets
- Memory usage monitoring during training
- Training loss and validation metrics tracking

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the transformer architecture
- [Salesforce CodeT5](https://huggingface.co/Salesforce/codet5-base) for the pre-trained model
- The open-source community for datasets and tools

## 📚 References

- [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/abs/2109.00859)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---
