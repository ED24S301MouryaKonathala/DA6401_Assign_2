import wandb
import torch
import torch.nn as nn
from model import CustomCNN
from dataset import get_dataloaders
import os

config = {
    'activation': 'relu',
    'batch_size': 32,
    'dense_neurons': 512,
    'dropout': 0.2,
    'filter_organization': 'same',
    'filter_size': 5,
    'num_filters': 64,
    'use_batchnorm': True,
    'use_dropout': True,
    'use_augmentation': True,
    'dataset_path': os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nature_12K'))
}

def test_retrained_model():
    with wandb.init(project="Assign_2_Part_A_iNaturalist_CNN_Final", name="sweep_retrained_test", config=config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get test dataloader
        _, _, test_loader = get_dataloaders(
            batch_size=config['batch_size'],
            dataset_path=config['dataset_path'],
            test_mode=True
        )
        
        # Setup model
        class_names = test_loader.dataset.classes
        model_config = config.copy()
        model_config['num_classes'] = len(class_names)
        
        # Load retrained model
        model = CustomCNN(model_config).to(device)
        model.load_state_dict(torch.load("sweep_best_model_retrained.pth"))
        print("Loaded retrained sweep model")
        
        # Test evaluation code same as before
        criterion = nn.CrossEntropyLoss()
        model.eval()
        
        test_loss = 0
        test_correct = 0
        total_test = 0
        all_preds = []
        all_labels = []
        test_images = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                total_test += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                test_images.extend(data.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Test Batch [{batch_idx+1}/{len(test_loader)}]")
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / total_test
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        wandb.summary.update({
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        
        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })
        
        # Create prediction grid
        images_list = []
        for idx, (img, pred, label) in enumerate(zip(test_images[:30], all_preds[:30], all_labels[:30])):
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(device))
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence = probs[0][pred].item() * 100
            
            status = "✓" if pred == label else "❌"
            caption = f"#{idx+1} | {status}\nPred: {class_names[pred]}\nTrue: {class_names[label]}\nConf: {confidence:.1f}%"
            
            images_list.append(wandb.Image(
                img.permute(1, 2, 0).numpy(),
                caption=caption
            ))
        
        wandb.log({
            "prediction_grid": images_list,
            "test_grid_metadata": {
                "layout": {"width": 3, "height": 10},
                "total_samples": 30,
                "test_accuracy": f"{test_acc:.2f}%"
            }
        })

if __name__ == "__main__":
    wandb.login()
    test_retrained_model()
