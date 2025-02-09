import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from config import Config
from utils import set_seed

# switch to right model for testing
if Config.MODEL == "CNNBase":
    from models.cnn_base import CNNModelBase as CNNModel
elif Config.MODEL == "CNNDeep":
    from models.cnn_deep import CNNModelDeep as CNNModel
elif Config.MODEL == "CNNDeepWithSE":
    from models.cnn_deep_with_se import CNNModelWithSEBlock as CNNModel
elif Config.MODEL == "CNNDeepWithCSE":
    from models.cnn_deep_with_cse import CNNModelWithConvSEBlock as CNNModel

if Config.DEVICE == "cuda":
    scaler = torch.amp.GradScaler()

saved_model_path = Config.SAVED_MODELS_PATH
saved_metrics_path = Config.SAVED_METRICS_PATH

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    running_loss = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if Config.DEVICE == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_loss = total_loss / (batch_idx + 1)  # Calculate running average
        # Update tqdm postfix with running statistics
        pbar.set_postfix(
            {"loss": f"{running_loss:.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(
                {
                    "val_loss": f"{running_loss:.4f}",
                    "val_acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    return total_loss / len(dataloader), 100.0 * correct / total

# CANDIDATE:
# Level 1. Create and use smaller dataset for faster training and smaller validation set for quick testing
# Level 2. Merge two spectator classes into one
# Level 3. Apply different transforms (which make sense) to the training and validation dataset (data augumentation). 
    # Apply them wisely for under-represented classes
def main():
    set_seed(Config.SEED)
    # Create dataset and dataloader
    global_dataset = ImageFolder(root=Config.DATA_PATH_TRAIN, transform=None)
    
    set_seed(Config.SEED)
    
   # Create dataset and dataloader
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ) # kinda like a default transformation

    # Merge 'spectators_long' and 'spectators_short' into 'spectators'
    class_map = {
        'closeup_head': 1,
        'closeup_waist': 2,
        'long': 3,
        'neg': 4,
        'short_player': 5,
        'spectators': 6,  # Merged class
    }
    class_to_idx = global_dataset.class_to_idx  # Mapping from class name to index
    idx_to_class = {value: key for key, value in class_to_idx.items()} 
    
    # Modify dataset labels
    def update_labels(dataset):
        # dataset.imgs: List of (image path, class_index) tuples
        updated_data = []
        for img_path, label in dataset.imgs:
            class_name = idx_to_class[label]
            if class_name in ['spectators_long', 'spectators_short']:
                new_idx = class_map['spectators']
            else:
                new_idx = class_map[class_name]
            
            updated_data.append((img_path, new_idx))
        
        dataset.imgs = updated_data
        return dataset

    # Modify dataset targets
    def update_target(dataset):
        new_targets = []
        for label in global_dataset.targets:
            class_name = idx_to_class[label] 
            if class_name in ['spectators_long', 'spectators_short']:
                new_idx = class_map['spectators']
            else:
                new_idx = class_map[class_name]
            new_targets.append(new_idx)
      
        dataset.targets = new_targets
        return dataset

    # Apply the label & target updates
    global_dataset = update_labels(global_dataset)
    global_dataset = update_target(global_dataset)
    
    dataset_size = len(global_dataset)
    reduction_factor = Config.DATA_REDUCTION_FACTOR
    small_dataset_size = int(reduction_factor * dataset_size)  # Smaller training dataset for faster training
    train_size = int(0.8 * small_dataset_size) # keeping 80-20% partition for train-val data
    val_size = small_dataset_size - train_size
    
    indices = random.sample(range(dataset_size), small_dataset_size) # select indices uniformly at random
    subset_global_dataset = Subset(global_dataset, indices) # uniformly select a subset of data
    print(f"Subset data size for training & tuning: {len(subset_dataset)}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        subset_global_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )
    
    train_dataset.dataset.transform = transform_train  # Apply different transforms to training
    val_dataset.dataset.transform = transform_val  # Keep validation set simple
    
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize model
    model = CNNModel(len(Config.CLASSES_ORIG))
    model = model.to(Config.DEVICE)

    # Define loss and optimizer
    # Experiment with different loss functions
    criterions = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "NLLLoss": nn.NLLLoss(),
        "MSELoss": nn.MSELoss(),
        "FocalLoss": FocalLoss()
    }
    criterion = criterions[Config.LOSS_FUNCTION]
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = StepLR(
        optimizer,
        step_size=Config.LR_STEP_SIZE,
        gamma=Config.LR_GAMMA,
    )

    # Training loop
    # CANDIDATE:
    # Level 1. Implement early stopping based on validation accuracy
    # Level 2. Use tensorboard for visualization (show also in video)
    # Level 3. Evaluate 3 different criterion functions
    
    # Training loop with early stopping, tensorboard, and multiple criterion evaluation
    writer = SummaryWriter(saved_metrics_path)
    best_val_acc = 0.0
    early_stopping_patience = 5
    no_improvement = 0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Step the scheduler
        if isinstance(scheduler, StepLR):
            scheduler.step()
        
        # Log results to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            os.makedirs(saved_model_path[:-1], exist_ok=True)
            model_type = Config.MODEL.lower()
            torch.save(model.state_dict(), saved_model_path+"best_model_"+model_type+".pth")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stopping_patience:
            print("Early stopping triggered!")
            break

        writer.close()


if __name__ == "__main__":
    main()
