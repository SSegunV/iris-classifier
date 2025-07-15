import argparse
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate(test_size, random_state):
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cMatrix = confusion_matrix(y_test, y_pred)

    return accuracy, cMatrix, iris.target_names, model

def save_confusion_matrix(conf_matrix, labels, output_dir):
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save image
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def save_model(model, output_dir):
    model_path = os.path.join(output_dir, "iris_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main(test_size, random_state):
    # Set up output directory in parent folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Train, evaluate and save
    accuracy, cMatrix, labels, model = train_and_evaluate(test_size, random_state)
    print(f"Accuracy: {accuracy:.2f}")

    save_confusion_matrix(cMatrix, labels, output_dir)
    save_model(model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use as test set.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state)
