# Multi-Layer Perceptron for New York Housing Market

## Project Description

In this project, I implemented a multi-layer perceptron (MLP) from scratch to solve a classification task using real-world data from the New York housing market. The goal was to predict the number of bedrooms for a given property based on various features such as price, type, square footage, and location details.

## Implementation Details

I followed these steps to complete the assignment:

1. **Data Preprocessing**: I cleaned and prepared the dataset, handling both numerical and categorical features. I used techniques such as one-hot encoding for categorical variables and normalization for numerical variables.

2. **Model Construction**: I built a vanilla feed-forward neural network with multiple hidden layers. The network's architecture, including the number of layers, nodes per layer, activation functions, and learning rate, was carefully chosen and tuned.

3. **Training the Model**: I trained the MLP using mini-batches and implemented backpropagation with gradient descent to update the weights. I also used techniques such as weight initialization and regularization to improve model performance and prevent overfitting.

4. **Hyperparameter Tuning**: I experimented with various hyperparameters, including the number of epochs, learning rate, batch size, and network architecture. I documented these experiments and their results in a `report.pdf`.

5. **Model Evaluation**: I evaluated the model's performance on multiple train/test splits provided in the dataset. The model's accuracy was compared to a baseline model, and I ensured it met the required performance thresholds.

6. **Prediction and Output**: After training, I used the trained model to predict the number of bedrooms for the test data. The predictions were saved in the required format in an `output.csv` file.

## Conclusion

This project involved implementing a neural network from scratch, handling various data preprocessing tasks, tuning hyperparameters, and ensuring efficient model training and evaluation. The final model successfully predicted the number of bedrooms with a performance that met or exceeded the baseline model's accuracy.
