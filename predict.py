import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
import os

from DA_ConvLSTM import MineralCNNLSTM_Flatten
from dataset_test import SpectraDataset

wavelet_name = 'cmor1-1.5'
scales = np.arange(3, 60)
num_classes = 3


def calculate_overall_precision(actuals, predictions):
    FN = np.sum((actuals != 0) & (predictions == 0))
    FP = np.sum((actuals == 0) & (predictions != 0))
    TN = np.sum((actuals == 0) & (predictions == 0))
    TN_25 = np.sum((actuals == 0) & (predictions < 10))
    TP = np.sum((actuals != 0) & (predictions != 0))
    TP_25 = np.sum((actuals != 0) & (predictions >= 10))
    total_accurate = TP + TN
    total_accurate_25 = TP_25 + TN_25
    overall_precision = total_accurate / len(actuals)
    overall_precision_25 = total_accurate_25 / len(actuals)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return overall_precision_25, overall_precision, TPR, FPR


def calculate_r2_rmse_for_non_zeros(actuals, predictions):
    non_zero_mask = (actuals != 0) & (predictions != 0)
    actuals_non_zero = actuals[non_zero_mask]
    predictions_non_zero = predictions[non_zero_mask]

    r2 = r2_score(actuals_non_zero, predictions_non_zero)
    rmse = np.sqrt(mean_squared_error(actuals_non_zero, predictions_non_zero))

    return r2, rmse


def error_plot(actuals, predictions, mineral_components):
    non_zero_mask = (actuals != 0) & (predictions != 0)
    actuals_non_zero = actuals[non_zero_mask]
    predictions_non_zero = predictions[non_zero_mask]

    errors_corrected = predictions_non_zero - actuals_non_zero
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals_non_zero, errors_corrected, color="red", alpha=0.5,
                label=f"{mineral_components} Error")
    plt.axhline(y=0, color='black', linestyle='--')

    plt.title(f"Error Scatter Plot for Mineral: {mineral_components}")
    plt.xlabel("Actual")
    plt.ylabel("Prediction Error")
    plt.grid(True)
    plt.ylim(-50, 50)
    plt.legend()
    plt.show()


def error_plot_line(actuals, predictions, mineral_components):
    non_zero_mask = (actuals != 0) & (predictions != 0)
    actuals_non_zero = actuals[non_zero_mask]
    predictions_non_zero = predictions[non_zero_mask]

    # Create figure and scatter plot
    plt.figure(figsize=(8, 6))
    #    plt.scatter(actuals_non_zero, predictions_non_zero, color="red", alpha=0.5,
    #                label=f"{mineral_components} Error")
    plt.plot([0, 100], [0, 100], color='black', linestyle='--')

    # Adding a boxplot
    boxplot_data = [
        predictions[(actuals == 10)],
        predictions[(actuals == 20)],
        predictions[(actuals == 30)],
        predictions[(actuals == 40)],
        predictions[(actuals == 50)],
        predictions[(actuals == 60)],
        predictions[(actuals == 70)],
        predictions[(actuals == 80)],
        predictions[(actuals == 90)]
    ]
    plt.boxplot(boxplot_data, positions=range(10, 100, 10), widths=4)

    # Set plot details
    plt.title(f"Actual vs Predicted Values for Mineral: {mineral_components}")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Removing grid lines
    plt.grid(False)
    # Saving the plot as an SVG file
    svg_filename_no_grid = './plot/' + mineral_components + '.svg'
    plt.savefig(svg_filename_no_grid, format='svg')

    plt.show()
    plt.close()


device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

# test data dir
folder_path = './test/test_data'
all_files = os.listdir(folder_path)

all_data = []

temp_df = pd.read_csv('data/mixed_spectra/sd_an.csv')

file_names = []
column = temp_df.columns

for file in all_files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, names=column)
        df = df.iloc[1:]
        all_data.append(df)
        file_names.extend([file] * len(df))

test_data = pd.concat(all_data, ignore_index=True)
# test_data['en'] = 0

spectra_test = test_data.iloc[:, :-4].values.astype(np.float64)
target_test = test_data.iloc[:, -4:].values.astype(np.float64)

test_dataset = SpectraDataset(spectra_test, target_test, num_classes)
test_dataloader = DataLoader(test_dataset, batch_size=300, shuffle=False)

model_fo = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1).to(device)
model_fo.load_state_dict(torch.load('./model/mineral_fo/mineral_fo_da_convlstm_best.pth'))
model_fo.eval()
with torch.no_grad():
    predictions_fo = []
    actuals_fo = []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        out, out1 = model_fo(inputs)
        out1 = torch.relu(out1)

        _, predicted_class = torch.max(out, dim=1)

        is_fo = predicted_class == 1
        predicted_quant = out1.squeeze() * is_fo.float()

        predictions_fo.append(predicted_quant.cpu().numpy())
        actuals_fo.append(labels[:, 0].cpu().numpy())

predictions_fo = np.concatenate(predictions_fo)
actuals_fo = np.concatenate(actuals_fo)

accuracy_fo_25, accuracy_fo, TPR_fo, FPR_fo = calculate_overall_precision(actuals_fo, predictions_fo)
r2_diff, rmse_diff = calculate_r2_rmse_for_non_zeros(actuals_fo, predictions_fo)

print(
    f"Overall precision 25%: {accuracy_fo_25:.4f};Overall precision: {accuracy_fo:.4f};TPR:{TPR_fo:.4f};FPR:{FPR_fo:.4f}")
print(f"R2 difference between actual and predicted values: {r2_diff}")
print(f"RMSE difference between actual and predicted values: {rmse_diff}")

# error_plot(actuals_fo, predictions_fo, 'fo')
# error_plot_line(actuals_fo, predictions_fo, 'fo')

model_aug = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1).to(device)
model_aug.load_state_dict(torch.load('./model/mineral_aug/mineral_aug_da_convlstm_best.pth'))
model_aug.eval()
with torch.no_grad():
    predictions_aug = []
    actuals_aug = []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        out, out1 = model_aug(inputs)
        out1 = torch.relu(out1)

        _, predicted_class = torch.max(out, dim=1)

        is_fo = predicted_class == 1
        predicted_quant = out1.squeeze() * is_fo.float()

        predictions_aug.append(predicted_quant.cpu().numpy())
        actuals_aug.append(labels[:, 1].cpu().numpy())

predictions_aug = np.concatenate(predictions_aug)
actuals_aug = np.concatenate(actuals_aug)

accuracy_aug_25, accuracy_aug, TPR_aug, FPR_aug = calculate_overall_precision(actuals_aug, predictions_aug)
r2_diff, rmse_diff = calculate_r2_rmse_for_non_zeros(actuals_aug, predictions_aug)

print(
    f"Overall precision 25%: {accuracy_aug_25:.4f};Overall precision: {accuracy_aug:.4f};TPR:{TPR_aug:.4f};FPR:{FPR_aug:.4f}")
print(f"R2 difference between actual and predicted values: {r2_diff}")
print(f"RMSE difference between actual and predicted values: {rmse_diff}")

# error_plot(actuals_aug, predictions_aug, 'aug')
# error_plot_line(actuals_aug, predictions_aug, 'aug')

model_an = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1).to(device)
model_an.load_state_dict(torch.load('./model/mineral_an/mineral_an_da_convlstm_best.pth'))
model_an.eval()
with torch.no_grad():
    predictions_an = []
    actuals_an = []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        out, out1 = model_an(inputs)
        out1 = torch.relu(out1)

        _, predicted_class = torch.max(out, dim=1)

        is_fo = predicted_class == 1

        predicted_quant = out1.squeeze() * is_fo.float()

        predictions_an.append(predicted_quant.cpu().numpy())
        actuals_an.append(labels[:, 2].cpu().numpy())

predictions_an = np.concatenate(predictions_an)
actuals_an = np.concatenate(actuals_an)

actuals_an_25, accuracy_an, TPR_an, FPR_an = calculate_overall_precision(actuals_an, predictions_an)
r2_diff, rmse_diff = calculate_r2_rmse_for_non_zeros(actuals_an, predictions_an)

print(
    f"Overall precision 25%: {actuals_an_25:.4f};Overall precision: {accuracy_an:.4f};TPR:{TPR_an:.4f};FPR:{FPR_an:.4f}")
print(f"R2 difference between actual and predicted values: {r2_diff}")
print(f"RMSE difference between actual and predicted values: {rmse_diff}")

# error_plot(actuals_an, predictions_an, 'an')
# error_plot_line(actuals_an, predictions_an, 'an')

model_en = MineralCNNLSTM_Flatten(num_classes=2, hidden_size=1024, num_layers=1).to(device)
model_en.load_state_dict(torch.load('./model/mineral_en/mineral_en_da_convlstm_best.pth'))
model_en.eval()
with torch.no_grad():
    predictions_en = []
    actuals_en = []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        out, out1 = model_en(inputs)
        out1 = torch.relu(out1)

        _, predicted_class = torch.max(out, dim=1)
        is_fo = predicted_class == 1

        predicted_quant = out1.squeeze() * is_fo.float()

        predictions_en.append(predicted_quant.cpu().numpy())
        actuals_en.append(labels[:, 3].cpu().numpy())

predictions_en = np.concatenate(predictions_en)
actuals_en = np.concatenate(actuals_en)

accuracy_en_25, accuracy_en, TPR_en, FPR_en = calculate_overall_precision(actuals_en, predictions_en)
r2_diff, rmse_diff = calculate_r2_rmse_for_non_zeros(actuals_en, predictions_en)

print(
    f"Overall precision 25%: {accuracy_en_25:.4f};Overall precision: {accuracy_en:.4f};TPR:{TPR_en:.4f};FPR:{FPR_en:.4f}")
print(f"R2 difference between actual and predicted values: {r2_diff}")
print(f"RMSE difference between actual and predicted values: {rmse_diff}")

# error_plot(actuals_en, predictions_en, 'en')
# error_plot_line(actuals_en, predictions_en, 'en')
