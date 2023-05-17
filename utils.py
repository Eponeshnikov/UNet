import torch
import torch.nn as nn
from torchmetrics import Dice


def check_accuracy(loader, model, DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    dice = Dice(average='macro', num_classes=23).to(DEVICE)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice(preds, y)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()
    return {'acc': num_correct / num_pixels * 100, 'dice': dice_score / len(loader)}
