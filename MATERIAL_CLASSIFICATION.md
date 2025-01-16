# Material Classification Based on Energy Gap and Semiconductor Types

This repository demonstrates material classification based on energy gaps and includes an explanation of **n-type** and **p-type** semiconductors. The project uses Python for classification and visualization.

## Theory

### Energy Gap Classification
The energy gap (bandgap) of a material determines whether it is a conductor, semiconductor, or insulator:

| **Material**      | **Energy Gap (eV)** | **Type**              |
|--------------------|---------------------|-----------------------|
| Conductor         | ≈ 0                | High conductivity      |
| Semiconductor     | 0.5 - 3            | Moderate conductivity  |
| Insulator         | > 3                | Low conductivity       |

### n-Type Semiconductor
- Doped with donor impurities (e.g., phosphorus in silicon).
- Provides extra electrons, increasing electron concentration.
- Electrons are the majority carriers; holes are the minority carriers.

### p-Type Semiconductor
- Doped with acceptor impurities (e.g., boron in silicon).
- Creates holes by accepting electrons, increasing hole concentration.
- Holes are the majority carriers; electrons are the minority carriers.

---

## Python Code Example for Material Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Sample data for material classification
data = np.array([
    [0.1, 0],  # Conductor
    [1.1, 1],  # Semiconductor
    [4.5, 2],  # Insulator
    [0.2, 0],  # Conductor
    [2.0, 1],  # Semiconductor
    [5.0, 2]   # Insulator
])

# Features: Energy Gap
X = data[:, 0].reshape(-1, 1)
# Labels: 0 = Conductor, 1 = Semiconductor, 2 = Insulator
y = data[:, 1]

# Create and train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Classify materials based on energy gap
test_data = np.array([0.3, 1.5, 3.2, 6.0]).reshape(-1, 1)
predictions = model.predict(test_data)

# Print classifications
categories = {0: 'Conductor', 1: 'Semiconductor', 2: 'Insulator'}
for gap, pred in zip(test_data.flatten(), predictions):
    print(f"Energy Gap: {gap} eV -> Material Type: {categories[pred]}")

# Visualization
plt.scatter(X, y, color='blue', label='Training Data')
plt.scatter(test_data, predictions, color='red', label='Predictions')
plt.xlabel("Energy Gap (eV)")
plt.ylabel("Material Type")
plt.legend()
plt.title("Material Classification Based on Energy Gap")
plt.show()

