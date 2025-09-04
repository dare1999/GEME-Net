# GEME-Net
This project implements a passenger flow prediction task based on knowledge distillation. The project includes two main models:
<img width="816" height="549" alt="image" src="https://github.com/user-attachments/assets/8f18057e-041e-4cb9-a699-a82c8919aa00" />

Teacher Model: a pre-trained large model that generates soft labels for prediction.

Student Model: a lightweight model that learns by mimicking the teacher model’s outputs, thereby achieving good performance with fewer computational resources.

Both the teacher and student models share the same data preprocessing and loading workflow. The data is stored in CSV format and loaded via a dedicated data_loader module. Inside the student model, feature extraction is carried out using an MLP Mixer combined with pointwise convolution. Some key parameters (e.g., output channels) are automatically inferred via a dummy forward pass to reduce hard coding.

Key File Structure

config.py
Configuration file. Defines device selection (GPU/MPS/CPU), data file paths (e.g., passenger flow, timetable, graph data), and data loading parameters (batch_size, window_size, step_size, train_ratio, label_dim). It also includes the ModelConfig class (covering in_channels, number of regions, event feature dimensions, attention parameters, etc.).

data_loader.py
Data loading module. Builds training, validation, and test datasets from CSV files, and generates auxiliary data such as adjacency matrices.

ModelFrame.py
Implementation of the teacher model. This pre-trained model generates soft labels for knowledge distillation.

student_model/models.py
Definition of the student model. Built from MLPMixer modules and pointwise convolution layers, the model outputs predictions (e.g., for 18 regions) via global average pooling and a linear layer. Key channel dimensions are inferred automatically through a dummy forward pass to reduce hard coding.

train.py
Training script for the teacher model. Uses 5-fold cross-validation and records training/validation metrics (MSE, RMSE, MAE, WMAPE), saving checkpoints during training.

val.py
Validation and visualization script for the teacher model. Loads checkpoints to plot training/validation loss curves, metric trends, and bar charts of the final epoch.

train_student_model.py
Training script for the student model. Uses teacher-generated soft labels and trains the student model with masked MAE loss, outputting detailed batch and epoch losses. The best model weights are saved in output/student_best.pth.

val_student.py
Validation and error visualization script for the student model. Loads trained student weights, computes absolute errors between teacher and student predictions on the validation set, and generates histograms, scatter plots, and sample error curves. All visualizations are saved in the pictures folder.

test_model.py
Compares teacher model, student model, and ground truth data through visualization.

How to Run

Create a virtual environment and install the required Python libraries.

Run train.py to train the teacher model; results are saved in the checkpoint folder.

Run train_student_model.py to train the student model; results are saved in the output folder.

Environment Requirements

Python 3.7+

PyTorch 1.8+

NumPy

matplotlib

scikit-learn

torch

others as needed

It is recommended to run the program on a GPU-enabled device (CUDA environment) for higher training efficiency.

Data Preprocessing

The raw input data has the shape [B, T, N] (B: batch size, T: time steps, N: number of nodes). After loading, the data is reshaped by the safe_reshape_for_mixer function into [B, in_channels, H, W], where:

in_channels is set according to the teacher model configuration (set to 1 in this project).

H and W are automatically inferred from the data to satisfy T × N = in_channels × H × W, ensuring compatibility with the MLPMixer input format.

Knowledge Distillation and Loss Definition
Knowledge Distillation

During training, the student model learns by minimizing the difference between its own predictions and the teacher model’s soft labels. The teacher model generates soft targets, and the student model iteratively reduces prediction errors relative to these labels. This process enables the student model to inherit knowledge from the teacher while remaining lightweight and efficient.
