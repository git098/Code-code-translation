import argparse
import torch
import evaluate
import os
import numpy as np
from transformers import (
    RobertaTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from data_handler import get_datasets

def main(args):
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['OMP_NUM_THREADS'] = '1'

    # Switch to debug dataset if debug flag is set
    if args.debug:
        args.data_dir = "./data/debug"
        args.percentage = 1.0

    train_dataset, valid_dataset, _ = get_datasets(args.data_dir, args.percentage)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        args.model_name, args.model_name
    )

    # Correct decoder start/end token IDs
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        inputs = tokenizer(
            examples['java_code'], padding='max_length', truncation=True, max_length=args.max_length
        )
        labels = tokenizer(
            examples['cs_code'], padding='max_length', truncation=True, max_length=args.max_length
        )
        # Mask pad tokens in labels
        label_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels['input_ids']
        ]
        inputs['labels'] = label_ids
        return inputs

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 back to pad_token_id before decoding labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[label] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing the data files.")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-model", help="Output directory for saving model checkpoints.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for storing logs.")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="Name of the base model to use.")
    parser.add_argument("--percentage", type=float, default=0.01, help="Percentage of the dataset to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every N steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps interval.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model checkpoint after every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Keep only the last N checkpoints.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")  # Increased default
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    main(args)
