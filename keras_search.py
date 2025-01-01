import pandas as pd
from keras_tuner import Hyperband
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import json
from sklearn.metrics import mean_squared_error, r2_score

# Load the main dataset
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

# Load the column names from the Excel files (assuming they are in the first row)
padel_cols = pd.read_excel('Padel_cols.xlsx', header=None).iloc[0].dropna().tolist()
spartan_cols = pd.read_excel('Spartan_cols.xlsx', header=None).iloc[0].dropna().tolist()
swissadme_cols = pd.read_excel('Swissadme_cols.xlsx', header=None).iloc[0].dropna().tolist()

def check_nans(data):
    rows_with_nans = data.isnull().any(axis=1)
    num_rows_with_nans = rows_with_nans.sum()
    total_rows = len(data)
    fraction_rows_with_nans = num_rows_with_nans / total_rows

    print(f"Number of rows with NaNs: {num_rows_with_nans}")
    print(f"Fraction of rows with NaNs: {fraction_rows_with_nans:.2f}")

    # Identify columns with NaNs
    columns_with_nans = data.columns[data.isnull().any()].tolist()
    num_columns_with_nans = len(columns_with_nans)

    print(f"Columns with NaNs: {columns_with_nans}")

    # Print detailed information about NaNs in each column
    nan_info = data.isnull().sum()
    print("\nDetailed NaN information:")
    print(nan_info[nan_info > 0])
    print("Number of columns with nan values:")
    print(f"{num_columns_with_nans} columns out of the total {len(data.columns)} columns")

def perform_pca(X_subset, explained_variance_threshold=0.95):
    # Print data nans before standardization
    print("NANs before standardization")
    check_nans(X_subset)

    # Drop zero variance columns
    zero_variance_columns = X_subset.loc[:, X_subset.std() == 0].columns
    X_subset = X_subset.drop(columns=zero_variance_columns)
    
    # Standardize the data if necessary
    X_standardized = (X_subset - X_subset.mean()) / X_subset.std()

    # Print data nans before standardization
    print("NANs after standardization")
    check_nans(X_standardized)

    # Initialize PCA
    pca = PCA()

    # Fit PCA
    pca.fit(X_standardized)

    # Calculate cumulative explained variance
    cum_var_explained = np.cumsum(pca.explained_variance_ratio_)
    
    # Determine the number of components needed to reach the explained variance threshold
    num_components = np.argmax(cum_var_explained >= explained_variance_threshold) + 1
    
    # Apply PCA with the selected number of components
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_standardized)
    
    return X_pca, num_components, pca.explained_variance_ratio_

# Perform PCA on each set of columns
X_padel_pca, padel_n_components, padel_variance_ratio = perform_pca(X[padel_cols])
X_spartan_pca, spartan_n_components, spartan_variance_ratio = perform_pca(X[spartan_cols])
X_swissadme_pca, swissadme_n_components, swissadme_variance_ratio = perform_pca(X[swissadme_cols])

# Display the number of components retained
print(f"Padel: {padel_n_components} components retained")
print(f"Spartan: {spartan_n_components} components retained")
print(f"SwissADME: {swissadme_n_components} components retained")

# Convert PCA results to DataFrames with 'Primary ID' as the index
X_padel_pca_df = pd.DataFrame(X_padel_pca, index=X['Primary ID'])
X_spartan_pca_df = pd.DataFrame(X_spartan_pca, index=X['Primary ID'])
X_swissadme_pca_df = pd.DataFrame(X_swissadme_pca, index=X['Primary ID'])

# Ensure that the other features DataFrame is also indexed by 'Primary ID'
other_features_cols = X.columns.difference(padel_cols + spartan_cols + swissadme_cols)
X_other_features = X[other_features_cols].set_index('Primary ID')

# Ensure that the target DataFrame is also indexed by 'Primary ID'
y.set_index('Primary ID', inplace=True)

# Merge the PCA-transformed data back together with the rest of the features
X_final = pd.concat([X_padel_pca_df, X_spartan_pca_df, X_swissadme_pca_df, X_other_features], axis=1)

# Make sure all the column names are strings
X_final.columns = X_final.columns.astype(str)

# Verify the final DataFrame shape and columns
print(f"Final DataFrame shape: {X_final.shape}")
print(X_final)


# Step 1: Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()

    # First layer with input dimension specified
    model.add(Dense(
        units=hp.Int('units_0', min_value=32, max_value=4056, step=32),
        activation='relu',
        input_dim=X_train.shape[1]  # Specify the input dimension for the first layer
    ))
    
    # Tune the number of layers
    for i in range(hp.Int('num_layers', 4, 20)):  # Search between 4 and 20 layers
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=4056, step=32),  # Number of neurons in each layer
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))  # Tune dropout rate between 0.0 and 0.5
    
    model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer for regression
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])  # Tune the learning rate
        ),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    
    return model

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set up the Keras Tuner
tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='keras_tuner_results',
    project_name='multi_output_regression'
)

# Callback to save the best model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model3.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Perform hyperparameter search with Keras Tuner
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])


# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Save the best hyperparameters to a JSON file
best_hyperparameters_dict = best_hyperparameters.values
with open('best_hyperparameters3.json', 'w') as json_file:
    json.dump(best_hyperparameters_dict, json_file)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save('best_model3.keras')

# Load the best model
loaded_model = tf.keras.models.load_model('best_model3.keras')

# Load the hyperparameters from JSON
with open('best_hyperparameters3.json', 'r') as json_file:
    loaded_hyperparameters = json.load(json_file)


# Predict on the test set using the best model
y_pred = loaded_model.predict(X_test)

# Convert predictions and test set to DataFrame for consistency with your metric calculations
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
y_test_df = pd.DataFrame(y_test, columns=y_test.columns)

# Calculate metrics
mse = mean_squared_error(y_test_df, y_pred_df, multioutput='raw_values')  # MSE for each target
r2_per_target = r2_score(y_test_df, y_pred_df, multioutput='raw_values')  # R² for each target
r2_overall = r2_score(y_test_df, y_pred_df, multioutput='variance_weighted')  # Overall weighted R²

# Calculate variance or range of each target for relative performance
variance_targets = np.var(y_test_df, axis=0)  # Variance of each target in test set
range_targets = np.ptp(y_test_df, axis=0)  # Range (max-min) of each target in test set

# Print MSE, R² score, variance, and range for each target
for target_name, mse_value, r2_value, variance_value, range_value in zip(y_test_df.columns, mse, r2_per_target, variance_targets, range_targets):
    print(f"Target: {target_name}, Mean Squared Error: {mse_value}, R² Score: {r2_value}")
    print(f"Target: {target_name}, Variance: {variance_value}, Range: {range_value}")
    print(f"Relative MSE (MSE/Variance): {mse_value/variance_value if variance_value != 0 else 'Undefined'}")
    print(f"Relative MSE (MSE/Range): {mse_value/range_value if range_value != 0 else 'Undefined'}\n")

# Print overall R² score
print(f"\nOverall R^2 Score: {r2_overall}")