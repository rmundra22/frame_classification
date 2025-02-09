import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config


# switch to right model for testing
if Config.MODEL == "CNNBase":
    from models.cnn_base import CNNModelBase as CNNModel
elif Config.MODEL == "CNNDeep":
    from models.cnn_deep import CNNModelDeep as CNNModel
elif Config.MODEL == "CNNDeepWithSE":
    from models.cnn_deep_with_se import CNNModelWithSEBlock as CNNModel
elif Config.MODEL == "CNNDeepWithCSE":
    from models.cnn_deep_with_cse import CNNModelWithConvSEBlock as CNNModel
    
artifacts_path = Config.ARTIFACTS_PATH

# CANDIDATE:
# Level 1. Calculate test accuracy, and classification report (use sklearn.metrics for this). Fill the results dictionary with these metrics.
# Level 2a. Calculate confusion matrix (use sklearn.metrics for this). Fill the results dictionary with this matrix.
# Level 2b. Up to this point you've probably noticed, that, some data images are incorrectly labeled. Propose a strategy to automatically find these images and return a list of them to the user.
# Level 3. Display the classification report and confusion matrix in a visually appealing way (e.g., using seaborn or matplotlib for confusion matrix visualization).

def test(args):
    test_dataset = ImageFolder(
        root=Config.DATA_PATH_TEST,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    
    # Create DataLoaders for train and validation sets
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Don't shuffle the test dataset
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize the model
    model_type = Config.MODEL.lower()
    print("Working with model {}!".format(model_type))
    model = CNNModel(len(Config.CLASSES_ORIG))
    model.load_state_dict(torch.load(args.model_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    print("Starting evaluation ...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            # print(f"Outputs: {outputs}, Predicted: {predicted}, Actual: {labels}")

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
   
    print("Completed evaluation! Reporting metrics ...")
    # Calculate accuracy and classification report
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=Config.CLASSES_ORIG, output_dict=True)

    results = {
        'accuracy': accuracy,
        'classification_report': class_report,
    }
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    results['confusion_matrix'] = conf_matrix

    # Identify incorrectly labeled images
    incorrect_indices = np.where(all_preds != all_labels)[0]
    incorrect_images = [(test_dataset.imgs[index][0], all_labels[index], all_preds[index]) for index in incorrect_indices]
    # For each incorrect image, we store the path, actual label, and predicted label
    results['incorrect_images'] = incorrect_images
    print("Total incorrect images for {} are: {}".format(model_type, len(results['incorrect_images'])))

    # Display the classification report and confusion matrix
    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASSES_ORIG, yticklabels=Config.CLASSES_ORIG)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(artifacts_path+"confusion_metric_for_"+model_type+".png") 
    plt.show()
    
    visualize_last_layer_embeddings(model, test_loader)

    return results

def visualize_last_layer_embeddings(model, dataloader):
    model_type = Config.MODEL.lower()
    
    # Get embeddings
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, target in dataloader:
            output = model(inputs)
            embeddings.append(output)
            labels.append(target)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    # Dimensionality reduction (t-SNE)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting the embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='jet', alpha=0.5)

    # If you need to mark specific points (part 2b), define them here 
    # For example, using specific indices
    points_to_mark = [0, 1, 2]  # Change this as per your requirement for your specific points
    marked_points = embeddings_2d[points_to_mark]
    plt.scatter(marked_points[:, 0], marked_points[:, 1], color='red', s=100, label='Marked Points', edgecolor='k')

    plt.title('t-SNE Visualization of Last Layer Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Class Label')
    plt.savefig(artifacts_path+"tsne_for_last_layer_embedding_with_"+model_type+".png") 
    plt.show()
    return 


def main():
    # CANDIDATE:
    # Level 1. Create/train two models in training phase. Change this code to evaluate both of them and compare results.
    # Level 2. Find existing exemplary image model (the best in your opinion for this task) and compare it (like in point 1) with the best one you trained
    # Level 3. Visualize last layer embeddings for the best trained model (you may need to install additional dependencies here). Visualize them to the user and mark the points denoted in part 2b of the previous assignment.
    parser = argparse.ArgumentParser(
        description="Test a trained image classification model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/best_model.pth",
        help="Path to the trained model weights",
    )

    args = parser.parse_args()
    
    # run this with 2 config setups say (model CNNBase and CNNDeep)
    results = test(args)


if __name__ == "__main__":
    main()
