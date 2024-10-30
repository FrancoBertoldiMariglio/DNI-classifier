import torch
import os
from tools.utils import train_and_validate, test, classify, calculate_metrics, convert_to_serializable


def run(train_dataloader, val_dataloader, test_dataloader, model, criterion, optimizer, model_name, num_epochs=15):

    # Definir el dispositivo (GPU o CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    early_stopping_patience = 5

    base_checkpoint_path = '/checkpoints'

    os.makedirs(base_checkpoint_path, exist_ok=True)

    checkpoint_path = base_checkpoint_path + '/best_model' + model_name + '.pth'

    train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, device,
                       num_epochs, early_stopping_patience, checkpoint_path)

    model.eval()

    y_true, y_proba = test(model, test_dataloader, device)

    y_true, y_pred, y_proba_flat = classify(y_proba, y_true)

    accuracy, precision, recall, specificity, fpr, tpr, roc_auc = calculate_metrics(y_true, y_pred, y_proba_flat)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")

    return accuracy, precision, recall, specificity, fpr, tpr, roc_auc
