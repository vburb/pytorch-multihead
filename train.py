# pylint: disable=[unused-import]
import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, Trainer, TrainingArguments

from encoder_model import DataProcessor, EncoderModel, load_jsonl

# results = classification_report(true_labels, eval_labels, target_names = target_names, zero_division=np.nan, output_dict = True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions_main = np.argmax(logits[0], axis=1)
    predictions_intent = np.argmax(logits[1], axis=1)
    labels_main = labels[:, 0]
    labels_intent = labels[:, 1]
    return {
        "main_accuracy": accuracy_score(labels_main, predictions_main),
        "intent_accuracy": accuracy_score(labels_intent, predictions_intent),
        "main_precision": precision_score(labels_main, predictions_main, average="weighted", zero_division=np.nan),
        "intent_precision": precision_score(labels_intent, predictions_intent, average="weighted", zero_division=np.nan),
        "main_recall": recall_score(labels_main, predictions_main, average="weighted", zero_division=np.nan),
        "intent_recall": recall_score(labels_intent, predictions_intent, average="weighted", zero_division=np.nan),
        "main_f1": f1_score(labels_main, predictions_main, average="weighted", zero_division=np.nan),
        "intent_f1": f1_score(labels_intent, predictions_intent, average="weighted", zero_division=np.nan),
    }


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_name = "google-bert/bert-large-cased"
    batch_size = 64
    epochs = 75
    lr = 5e-6
    weight_decay = 0.01
    warmup_steps = 500

    tokenizer = BertTokenizer.from_pretrained(model_name)

    logger.info("Loading data and creating data loader")
    processor = DataProcessor(tokenizer=tokenizer, max_length=16)

    # data loading and data loader creation
    train = load_jsonl("./data/....jsonl")
    train = processor.build_dataset(train)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=False)

    validation = load_jsonl("./data/labeler-validation.jsonl")
    validation = processor.build_dataset(validation)
    validation_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    logger.info("Model initialization")
    # Model initialization
    config = BertConfig.from_pretrained(model_name)
    MODEL = EncoderModel(config).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        num_train_epochs=epochs,  # Total number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size per device during training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        warmup_steps=warmup_steps,  # Number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_strategy="epoch",  # Log every 10 steps
        # logging_steps=10,
        do_eval=True,
        evaluation_strategy="epoch",  # Evaluation is done at the end of each epoch
        report_to="tensorboard",  # Enable Tensorboard
        save_strategy="no",  # TODO Check these args to make it like training script
    )

    logger.info("Training model")
    # Initialize Trainer
    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=validation_dataloader.dataset,
        compute_metrics=compute_metrics,  # Define compute_metrics function for evaluation
    )

    # Start training
    trainer.train()

    # Evaluate the model
    # trainer.evaluate()

    logger.info("Saving model")
    # Save model to file
    MODEL.save_pretrained("./models/")


