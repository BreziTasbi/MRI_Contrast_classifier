from torch.utils.data import DataLoader
from data_manager import Dataset_2D, find_T1w_T2w_paths, dataset_splitter, Dataset_2D, paths_to_Dataset
from contrast_classifier_Networks import ResNet18SingleChannel, FCN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

# Define the training loop
def training_one_epoch(model):
    model.train()
    running_loss = 0.0
    queue_line = np.arange(train_dataset.length)
    np.random.shuffle(queue_line) 
    index=0
    for i in queue_line:
        if index%200 == 0:
            print(index)
        image, label = train_dataset[i]
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        print(outputs.shape, label.shape)
        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        index+=1
    return model, running_loss / len(train_dataset)

#### Load the data
# Define the base directory
base_dir = "data//data-multi-subject//"

# Find the relative paths of the T1w and T2w files in the specified directory
t1w_file_paths, t2w_file_paths = find_T1w_T2w_paths(base_dir)

# Split the data into training and validation sets
pd_train_data, pd_val_data = dataset_splitter(t1w_file_paths, t2w_file_paths)

# Create the training and validation datasets
train_dataset = paths_to_Dataset(pd_train_data)
val_dataset = paths_to_Dataset(pd_val_data)

#### Train the model

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = ResNet18SingleChannel(num_classes=2).to(device)
model = FCN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")
    model, train_loss = training_one_epoch(model)
    print(f"Epoch {epoch + 1} training loss: {train_loss}")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model_{epoch + 1}.pth")
#save model
torch.save(model.state_dict(), "model.pth")

#### Evaluate the model
evaluate = False

if evaluate:
    # Assess accuracy and F1 score on the validation set
    model.eval()
    val_predictions = []
    val_labels = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        image, label = image.to(device), label.to(device)
        output = model(image)
        prediction = torch.round(output)
        val_predictions.append(prediction.item())
        val_labels.append(label.item())

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)

    accuracy = np.mean(val_predictions == val_labels)
    f1_score = f1_score(val_labels, val_predictions)
