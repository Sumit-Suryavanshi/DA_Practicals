# 📦 Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 📥 Load the built-in Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])

print("🔍 Sample Data:")
print(df.head())

# 🎨 Visualizations - Sepal
df0 = df[df.target == 0]  # Setosa
df1 = df[df.target == 1]  # Versicolor
df2 = df[df.target == 2]  # Virginica

plt.figure(figsize=(8, 5))
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label="Setosa")
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label="Versicolor")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.legend()
plt.grid(True)
plt.show()

# 🎨 Visualizations - Petal
plt.figure(figsize=(8, 5))
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label="Setosa")
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label="Versicolor")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')
plt.legend()
plt.grid(True)
plt.show()

# 🔄 Prepare features & labels
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print("Train set size:", len(X_train))
print("Test set size:", len(X_test))

# 🤖 KNN Model Training
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# 🔮 Sample Prediction
sample = [[4.8, 3.0, 1.5, 0.3]]
predicted_class = knn.predict(sample)[0]
print(f"Prediction for {sample}: {iris.target_names[predicted_class]}")

# 📊 Model Evaluation
y_pred = knn.predict(X_test)

# 🧩 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 📝 Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
