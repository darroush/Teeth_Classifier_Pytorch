import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def dataset_loader(dataset_dir, transform, batch_size= 128):
    dataset = datasets.ImageFolder(root= dataset_dir, transform= transform)
    num_images = len(dataset)
    indices = list(range(num_images))
    random.seed(50)
    random.shuffle(indices)
    
    train_size = int(0.7 * num_images)
    val_size = int(0.2 * num_images)
    test_size = num_images - train_size - val_size
    
    train_indices, val_test_indices = indices[:train_size], indices[train_size:]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=test_size, random_state=3)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices) 
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return dataset, train_loader, val_loader, test_loader

def plot_loss_accuracy(epoch_log, loss_log, accuracy_log):
    fig, ax1 = plt.subplots()
    plt.title('Validation Accuracy & Loss vs Epoch')
    ax2 = ax1.twinx()
    ax1.plot(epoch_log, loss_log, 'go--')
    ax2.plot(epoch_log, accuracy_log, 'bo--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='g')
    ax2.set_ylabel('Val Accuracy', color='b')
    plt.show()
    
def evaluate_model(test_loader, net):
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    return f'Test Accuracy: {test_accuracy:.3f}%'

def plot_grid_view(dataset, net, test_loader):
    class_dict = dataset.class_to_idx
    reverse_class_dict = {v: k for k,v in class_dict.items()}
    
    fig, ax = plt.subplots(figsize=(12,15))
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)

    for i in range(32):
        plt.subplot(8,4,i+1)
        img = np.reshape(images[i].cpu().numpy(),[64,64])
        plt.imshow(img, cmap='gray')
        plt.title(
            f'Predicted Class: {reverse_class_dict[predicted[i].item()]} \n' 
            f'Ground Truth: {reverse_class_dict[labels[i].item()]}'
            )
    fig.tight_layout()
    plt.show()
    
def plot_cmatrix_creport(test_loader, net, dataset):
    pred_list = []
    label_list = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            pred_list.extend(predicted.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(label_list, pred_list)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels= dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(label_list, pred_list, target_names= dataset.classes))

    
    



if __name__ == '__main__':
    print('Utils is working')