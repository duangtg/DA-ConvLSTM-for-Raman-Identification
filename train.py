import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from DA_ConvLSTM import MineralCNNLSTM_Flatten
from dataread import dfbuilder
from dataset import SpectraDataset

learning_rate = 0.001
batch_size = 300
drop_rate = 0.55
epochs = 5
num_classes = 3

hyperparameters = [learning_rate, batch_size, drop_rate, epochs]

if torch.backends.mps.is_built():
    print("MPS ENVIRONMENT READY")

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

spectra_train_temp, spectra_test, labels_train_temp, labels_test = dfbuilder(fin_path="data/corrected_spectra",
                                                                             dev_size=0.05, r_state=25, raw=True,
                                                                             mineral_index=1)
spectra_train, spectra_val, labels_train, labels_val = train_test_split(spectra_train_temp, labels_train_temp,
                                                                        test_size=0.05, random_state=42)

# Create Dataset and DataLoader
train_dataset = SpectraDataset(spectra_train.values, labels_train.values, num_classes)
val_dataset = SpectraDataset(spectra_val.values, labels_val.values, num_classes)
test_dataset = SpectraDataset(spectra_test.values, labels_test.values, num_classes)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1).to(device)

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate adjustment strategy
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)

# Store loss and accuracy during training
train_losses = []
val_losses = []
val_mse_scores = []
val_rmse_scores = []
val_mae_scores = []
val_accuracies = []
val_r2 = []

criterion_re = nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()

# Set the epoch interval for saving the model
save_interval = 4

# training
num_epochs = 20
lambda_penalty = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out, out1 = model(inputs)
        target = labels[:, 1:2].long()
        target1 = labels[:, 0:1]

        loss2 = criterion_re(out1, target1)
        loss1 = criterion(out, target.squeeze())
        loss = lambda_penalty * loss1 + loss2
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Compute average loss
    average_loss = epoch_loss / len(train_dataloader)
    train_losses.append(average_loss)

    # Evaluate on the validation set
    model.eval().to(device)
    val_loss = 0.0
    total, correct = 0, 0
    mse, r2 = 0, 0

    y_pred_class = []
    y_true_class = []
    y_pred_regr = []
    y_true_regr = []

    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, labels in val_dataloader:
            # outputs = model(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            out, out1 = model(inputs)
            target = labels[:, 1:2].long()
            target1 = labels[:, 0:1]

            loss2 = criterion_re(out1, target1)
            loss1 = criterion(out, target.squeeze())
            loss = lambda_penalty * loss1 + loss2
            val_loss += loss.item()

            # for classification
            _, predicted_class = torch.max(out, 1)  # Get the index of the largest logit as the predicted category
            y_pred_class.extend(predicted_class.cpu().numpy())
            y_true_class.extend(target.cpu().numpy())

            # for regression
            y_pred_regr.extend(out1.view(-1).cpu().numpy())
            y_true_regr.extend(target1.view(-1).cpu().numpy())

    accuracy = accuracy_score(y_true_class, y_pred_class)
    mse = mean_squared_error(y_true_regr, y_pred_regr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_regr, y_pred_regr)

    val_accuracies.append(accuracy)
    val_rmse_scores.append(rmse)
    val_r2.append(r2)

    # Calculate average validation loss
    average_val_loss = val_loss / len(val_dataloader)
    val_losses.append(average_val_loss)

    # Print
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')

    if (epoch + 1) % save_interval == 0:
        save_path = f'./cnn_cwt/mineral_aug/mineral_cnn_lstm_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch + 1} to {save_path}")

    # Update learning rate
    lr_scheduler.step(average_val_loss)

epochs = range(1, num_epochs + 1)


# draw the training curve
# Create two side-by-side charts
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 6))

# One for accuracy and loss
ax2 = ax1.twinx()
ax1.plot(epochs, val_accuracies, color='red', label='Accuracy')
ax2.plot(epochs, val_losses, color='green', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='black')
ax2.set_ylabel('Loss', color='black')
ax1.set_title('Accuracy and Loss')

# one for RMSE and R2
ax4 = ax3.twinx()
ax3.plot(epochs, val_rmse_scores, color='black', label='RMSE')
ax4.plot(epochs, val_r2, color='blue', label='R2 Score')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('RMSE', color='black')
ax4.set_ylabel('R2 Score', color='black')
ax3.set_title('RMSE and R2 Score')

# save
plt.savefig('./plot/training_accuracy_loss.svg', format='svg')

plt.show()

