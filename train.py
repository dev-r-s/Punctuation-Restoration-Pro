import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report
import numpy as np
import json
from tqdm import tqdm
import pickle

from config import *
from dataset import PunctuationDataset


def calculate_class_weights(train_labels: list, num_labels: int) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    label_counts = np.zeros(num_labels)
    total_samples = 0
    
    for label_seq in train_labels:
        for label in label_seq:
            if label != "O":
                label_id = LABEL2ID[label]
                label_counts[label_id] += 1
            else:
                label_counts[0] += 1
            total_samples += 1
    
    # Avoid division by zero
    label_counts = np.maximum(label_counts, 1)
    
    # Calculate weights: N_samples / (N_classes * Count_class)
    weights = total_samples / (num_labels * label_counts)
    
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits  # (Batch, Seq_Len, Num_Labels)
            
            # Flatten for loss calculation
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, NUM_LABELS),
                labels.view(-1)
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return F1 scores."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Flatten and filter out -100 labels
            pred_flat = predictions.view(-1).cpu().numpy()
            label_flat = labels.view(-1).cpu().numpy()
            
            mask = label_flat != -100
            all_preds.extend(pred_flat[mask])
            all_labels.extend(label_flat[mask])
    
    # Calculate classification report
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return report


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model WITH add_prefix_space=True [FIXED HERE]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)
    
    # =================================================================
    # IMPORTANT: Replace this Dummy Data with your Real 100k Dataset
    # =================================================================
    print("WARNING: Using Dummy Data. Please load your real dataset here.")
    
    # Example of how you will load your data:
    # with open('my_dataset.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # train_texts = data['train_texts']
    # train_labels = data['train_labels']
    
    # DUMMY DATA
    train_texts = [
        "this is a sample sentence",
        "another example without punctuation",
        "how are you doing today"
    ] * 100
    
    train_labels = [
        ["O", "O", "O", "O", "PERIOD+CAPS"],
        ["PERIOD+CAPS", "O", "O", "PERIOD+CAPS"],
        ["O", "O", "O", "O", "QM+CAPS"]
    ] * 100
    # =================================================================

    
    # Create datasets
    train_dataset = PunctuationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        label2id=LABEL2ID,
        augment=True,
        lowercase_prob=RANDOM_LOWERCASE_PROB
    )
    
    eval_dataset = PunctuationDataset(
        texts=train_texts[:10],
        labels=train_labels[:10],
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        label2id=LABEL2ID,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )
    
    # Calculate class weights
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(train_labels, NUM_LABELS)
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Gradient scaler for FP16
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, class_weights
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        report = evaluate(model, eval_loader, device)
        
        # Print specific metrics
        print(f"\nQM F1-Score: {report.get('QM', {}).get('f1-score', 0.0):.4f}")
        print(f"PERIOD F1-Score: {report.get('PERIOD', {}).get('f1-score', 0.0):.4f}")
        print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        
        # Save best model
        current_f1 = report['macro avg']['f1-score']
        if current_f1 > best_f1:
            best_f1 = current_f1
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
            
            # Save config
            config_dict = {
                'MAX_LEN': MAX_LEN,
                'OVERLAP_STRIDE': OVERLAP_STRIDE,
                'LABEL2ID': LABEL2ID,
                'ID2LABEL': ID2LABEL,
                'NUM_LABELS': NUM_LABELS,
                'MODEL_NAME': MODEL_NAME
            }
            with open(f"{MODEL_OUTPUT_DIR}/config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"Model saved with F1: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
