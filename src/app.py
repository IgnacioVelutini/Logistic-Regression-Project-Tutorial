import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to load the dataset
def load_data():
    url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
    df = pd.read_csv(url)
    return df

# Function to preprocess the data (one-hot encoding)
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    return X, y

# Function to split the dataset into training and test sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to build and train the linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-Squared: {r2}")
    return y_pred

# Function to plot residuals
def plot_residuals(y_test, y_pred):
    sns.residplot(x=y_test, y=y_pred, lowess=True, color="g")
    plt.xlabel('Observed')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

# Function to save the model to a file
def save_model(model, filename="linear_regression_model.pkl"):
    joblib.dump(model, filename)

# Function to load a saved model
def load_model(filename="linear_regression_model.pkl"):
    return joblib.load(filename)

# Main execution function
def main():
    # Step 1: Load the dataset
    df = load_data()
    print("Dataset loaded successfully!")
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(df)
    print("Data preprocessing completed.")
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Step 4: Train the model
    model = train_model(X_train, y_train)
    print("Model training completed.")
    
    # Step 5: Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Step 6: Plot residuals (Optional)
    plot_residuals(y_test, y_pred)
    
    # Step 7: Save the model (Optional)
    save_model(model)
    print("Model saved as 'linear_regression_model.pkl'")

# Run the main function
if __name__ == "__main__":
    main()
