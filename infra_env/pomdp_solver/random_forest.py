from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib 


def random_forest():
    # Load the dataset
    X = np.load('data/training_input_data.npy')
    y = np.load('data/b_vals_all_components.npy')
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # Splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest regressor
    model = RandomForestRegressor(n_estimators=500, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Perform cross-validation to evaluate model performance
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validated MSE: {-np.mean(scores)}")

    # Determine "good" predictions with error less than a threshold
    error_threshold = 1e-4
    good_predictions = np.abs(y_test - y_pred) < error_threshold
    num_good_predictions = np.sum(good_predictions)
    print(f"Number of 'good' predictions (Error < {error_threshold} units): {num_good_predictions}; out of {len(y_test)} total predictions")

    # Save the model to a file
    model_filename = 'results/random_forest_regressor.joblib'
    joblib.dump(model, model_filename)

if __name__ == '__main__':
    random_forest()

