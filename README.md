# Aygaz-Deep-Learning-Project


# Dog Breed Identification

## Overview

This repository contains a comprehensive Python script for dog breed identification. The workflow includes data preprocessing, Convolutional Neural Network (CNN) model creation, training, evaluation, and integration with a Random Forest classifier for hyperparameter tuning.

## Setup

## Code Structure

- `Aygaz-Dog-Breed-Projem.ipynb`: The main Python script for the entire workflow.
- `best_model.h5`: Saved best-performing CNN model.
- `README.md`: Project documentation.

## Usage

The script performs the following steps:

1. **Data Loading and Preprocessing:**
   - Import necessary libraries and load dog breed information.
   - Resize images to 128x128 pixels and construct a DataFrame.

2. **Label Encoding and One-Hot Encoding:**
   - Encode the 'breed' column numerically.
   - Apply one-hot encoding and concatenate columns with the original DataFrame.

3. **Data Splitting and Normalization:**
   - Split the dataset into training, testing, and validation sets.
   - Normalize pixel values to the range [0, 1].

4. **Convolutional Neural Network (CNN) Model:**
   - Build a CNN model using Keras with appropriate layers.
   - Compile the model using the Adam optimizer and categorical cross-entropy loss.

5. **Model Training and Checkpointing:**
   - Train the model with a ModelCheckpoint callback for saving the best model.
   - Visualize training progress with accuracy and loss plots.

6. **Saving and Loading the Best Model:**
   - Save the best model as 'best_model_copy.h5'.
   - Load the best model for future use.

7. **Integration with Random Forest for Hyperparameter Tuning:**
   - Integrate the pre-trained CNN model with a Random Forest classifier using GridSearchCV.
   - Print the best hyperparameters and evaluate performance on the test set.

## Conclusion

This project provides a robust pipeline for dog breed identification, combining deep learning techniques with traditional machine learning methods. The script is designed for easy reproducibility and adaptability to similar classification tasks.

## Next Steps

Consider exploring additional augmentation techniques, fine-tuning the CNN architecture, and expanding the hyperparameter search space for further optimization. Regularly monitor and retrain the model to adapt to evolving data patterns.

## Acknowledgements

This work leverages the capabilities of TensorFlow, Keras, and scikit-learn. Special acknowledgment is given to the open-source community for their contributions to these libraries.
