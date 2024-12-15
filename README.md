# AutoML-GUI
AutoML GUI: A user-friendly Python tool to simplify machine learning workflows with a graphical interface. Load datasets, train models, and evaluate results with ease—no coding required

# Overview

AutoML GUI is a Python-based graphical user interface (GUI) tool designed to simplify the process of training and evaluating machine learning models. With just a few clicks, users can:

Load a dataset.

Specify the target column.

Train a Random Forest model.

View evaluation results (accuracy and classification report).

This project is beginner-friendly and serves as a foundational tool for anyone exploring AutoML concepts.

# Features

Dataset Loading: Load CSV datasets directly through the GUI.

Target Column Selection: Specify the column to predict (classification tasks).

Basic AutoML Workflow:

Data preprocessing.

Train/test split.

Training a Random Forest model.

Evaluation Results:

View model accuracy.

See a detailed classification report.

# Installation

To get started with AutoML GUI, follow these steps:

Prerequisites

* Python 3.7+

* pip package manager

# Dependencies

Install the required Python libraries:

pip install pandas scikit-learn

# How to Use

Run the Application:
Save the script as automl_gui.py and execute it:

python automl_gui.py

Load a Dataset:

Click the Load Dataset button.

Select a CSV file from your system.

Specify the Target Column:

Enter the name of the column you want to predict in the Target Column text field.

Run AutoML:

Click the Run AutoML button.

The tool will process your data, train a model, and display the evaluation results in the text area.

# Example Dataset

Here is an example of a simple dataset format:

feature1,feature2,feature3,target
1.2,0.8,3.1,ClassA
2.1,1.0,1.5,ClassB
0.5,2.3,0.8,ClassA

Target Column: target

# Contributing

Contributions are welcome! Feel free to:

Fork this repository.

Submit a pull request with improvements or new features.
