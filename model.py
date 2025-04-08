import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

# Define features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to be used
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "LogReg": LogisticRegression(max_iter=200),
    "Tree": DecisionTreeClassifier(max_depth=3),
    "SVM": SVC(kernel='linear'),
    "RF": RandomForestClassifier()
}

# Store results for comparison
results = []
for name, model in models.items():
    # Train each model
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy on the test set
    test_acc = accuracy_score(y_test, y_pred)

    # Perform cross-validation and calculate mean and standard deviation of accuracy
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Print the results for each model
    print(f"{name} : test acc = {test_acc:.2f} | CV acc = {cv_mean:.2f} ± {cv_std:.2f}")

    # Append the results to the list
    results.append({
        "Name": name,
        "Model": model,
        "Accuracy": test_acc,
        "CV Mean": cv_mean,
        "CV Std": cv_std
    })

# Find the best model based on test accuracy > CV mean > CV std
best_model = sorted(results, key=lambda x: (-x["Accuracy"], -x["CV Mean"], x["CV Std"]))[0]

# Visualization using only 2 features: sepal length & sepal width
X_vis = X.iloc[:, [0, 1]]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.2, random_state=42)
scaler_vis = StandardScaler()
X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
X_test_vis_scaled = scaler_vis.transform(X_test_vis)

# Train the best model again on the 2D data
model_2d = best_model["Model"].__class__(**best_model["Model"].get_params())
model_2d.fit(X_train_vis_scaled, y_train_vis)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_test_vis_scaled[:, 0].min() - 1, X_test_vis_scaled[:, 0].max() + 1
y_min, y_max = X_test_vis_scaled[:, 1].min() - 1, X_test_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the class for each point in the mesh grid
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
cmap = ListedColormap(['red', 'green', 'blue'])

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot the scatter points and add a label for the legend
scatter = plt.scatter(X_test_vis_scaled[:, 0], X_test_vis_scaled[:, 1],
            c=y_test_vis, cmap=cmap, edgecolor='k')

# Set the title and axis labels
plt.title(f'Decision Boundary - {best_model["Name"]} (acc: {best_model["Accuracy"]:.2f})')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Add the legend
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)

plt.show()

