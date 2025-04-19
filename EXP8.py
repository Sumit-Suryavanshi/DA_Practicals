import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and prepare the dataset
def load_and_prepare_data():
    # Load the Mall Customers dataset
    df = pd.read_csv('Mall_Customers.csv')
    
    # Create synthetic target variable - would buy insurance (1) or not (0)
    # Higher probability for older customers with higher income and moderate spending
    np.random.seed(42)
    prob = 1/(1 + np.exp(-(0.08*df['Age'] + 0.03*df['Annual Income (k$)'] - 0.01*abs(df['Spending Score (1-100)']-50) - 4))
    df['would_buy_insurance'] = np.random.binomial(1, prob)
    
    return df

# Visualize relationships
def plot_relationships(df):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df['Age'], df['would_buy_insurance'], alpha=0.5)
    plt.title('Age vs Insurance Purchase')
    plt.xlabel('Age')
    plt.ylabel('Insurance Purchase')
    
    plt.subplot(1, 3, 2)
    plt.scatter(df['Annual Income (k$)'], df['would_buy_insurance'], alpha=0.5)
    plt.title('Income vs Insurance Purchase')
    plt.xlabel('Annual Income (k$)')
    
    plt.subplot(1, 3, 3)
    plt.scatter(df['Spending Score (1-100)'], df['would_buy_insurance'], alpha=0.5)
    plt.title('Spending Score vs Insurance Purchase')
    plt.xlabel('Spending Score')
    
    plt.tight_layout()
    plt.show()

# Main prediction function
def predict_insurance_purchase():
    # Load and prepare data
    df = load_and_prepare_data()
    
    print("\n=== Dataset Overview ===")
    print(f"Total customers: {len(df)}")
    print(f"Customers who would buy insurance: {df['would_buy_insurance'].sum()} ({df['would_buy_insurance'].mean()*100:.1f}%)")
    print("\nFirst 5 records:")
    print(df.head())
    
    # Visualize relationships
    plot_relationships(df)
    
    # Prepare features and target
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    y = df['would_buy_insurance']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Prediction function for new customers
    def predict_new_customer(age, income, spending_score):
        customer_data = scaler.transform([[age, income, spending_score]])
        prob = model.predict_proba(customer_data)[0][1]
        prediction = model.predict(customer_data)[0]
        
        result = {
            'Age': age,
            'Income': income,
            'Spending Score': spending_score,
            'Prediction': 'Yes' if prediction == 1 else 'No',
            'Probability': f"{prob:.2%}",
            'Confidence': 'High' if prob > 0.7 or prob < 0.3 else 'Medium'
        }
        return result
    
    # Example predictions
    print("\n=== Example Predictions ===")
    examples = [
        (25, 40, 60),   # Young, moderate income, high spending
        (45, 80, 40),   # Middle-aged, good income, moderate spending
        (60, 120, 50),  # Older, high income, balanced spending
        (30, 30, 90)    # Young adult, low income, very high spending
    ]
    
    for example in examples:
        prediction = predict_new_customer(*example)
        print(f"\nCustomer (Age: {prediction['Age']}, Income: ${prediction['Income']}k, Spending: {prediction['Spending Score']}):")
        print(f"  - Would buy insurance: {prediction['Prediction']}")
        print(f"  - Probability: {prediction['Probability']} ({prediction['Confidence']} confidence)")

# Run the prediction
if __name__ == "__main__":
    predict_insurance_purchase()
