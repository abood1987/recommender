import torch
from django.core.management.base import BaseCommand
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


class Command(BaseCommand):
    help = "Fine-Tuning FLAN-T5 Large with the ESCO Database for Skill Extraction"
    requires_system_checks = False

    def handle(self, *args, **options):
        self.stdout.write("---START---")
        import os
        output_dir = "./flan_t5_esco"
        logging_dir = "./logs"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        MODEL_NAME = "google/flan-t5-base"

        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

        data = load_dataset("json", data_files="C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\dataset\\esco_skills.json")
        data = data["train"].train_test_split(test_size=0.2)
        print(data["train"][:5])

        prefix = "Extract skills from: "

        def preprocess_function(doc):
            """Add prefix to the sentences, tokenize the text, and set the labels"""
            # The "inputs" are the tokenized answer:
            inputs = [prefix + d for d in doc["input"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)

            # The "labels" are the tokenized outputs:
            labels = tokenizer(text_target=doc["output"], max_length=512, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = data.map(preprocess_function, batched=True)

        nltk.download("punkt", quiet=True)
        nltk.download('punkt_tab')
        metric = evaluate.load("rouge")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            # ✅ Ensure labels use pad_token_id instead of -100
            labels = np.array(labels)
            labels[labels == -100] = tokenizer.pad_token_id

            # ✅ Ensure preds have valid token IDs
            preds = np.array(preds)
            preds[preds < 0] = tokenizer.pad_token_id
            preds[preds >= tokenizer.vocab_size] = tokenizer.pad_token_id

            # ✅ Ensure preds and labels have the same shape
            if preds.shape != labels.shape:
                min_length = min(preds.shape[1], labels.shape[1])
                preds = preds[:, :min_length]
                labels = labels[:, :min_length]

            # ✅ Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # ✅ Format for Rouge metric
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

            # ✅ Compute Rouge Score
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

            return result

        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            eval_strategy="epoch",
            learning_rate=2e-5, # = 0.00002 → A small learning rate ensures gradual learning and prevents overfitting (Controls how much the model updates weights after each training step)
            per_device_train_batch_size=8, # Defines how many examples are processed per GPU (or CPU) during training,
            per_device_eval_batch_size=4, # Same as training batch size but used for validation/evaluation.
            weight_decay=0.01, # Prevents overfitting by applying L2 regularization to the model’s weights
            save_total_limit=3, # Limits the number of saved model checkpoints,
            num_train_epochs=3,
            predict_with_generate=True,
            push_to_hub=False,
        )

        torch.amp.autocast("cpu") # Fix: Ensure correct torch autocast
        model.config.use_cache = False # Fix: Prevent `past_key_values` warning

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        self.stdout.write("---END---")
