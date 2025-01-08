import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class IrisClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        
        from sklearn.datasets import load_iris
        data = load_iris()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

    def preprocess_data(self):
       
        X = self.df.drop('species', axis=1)
        y = self.df['species']

       
        X_scaled = self.scaler.fit_transform(X)

       
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def train_model(self):
       
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
       
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
       
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = self.scaler.transform(input_data)

       
        prediction = self.model.predict(input_scaled)
        return prediction[0]

def main():
    
    classifier = IrisClassifier()

    
    classifier.load_data()
    classifier.preprocess_data()

    
    classifier.train_model()

    
    classifier.evaluate_model()

    
    print("\n--- Predicting Iris Species ---")
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    prediction = classifier.predict(sepal_length, sepal_width, petal_length, petal_width)
    print(f"The predicted Iris species is: {prediction}")

if __name__ == "__main__":
    main()
