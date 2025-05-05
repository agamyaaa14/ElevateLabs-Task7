# ElevateLabs Internship
## 🧠 Task 7: Support Vector Machines (SVM) - Breast Cancer Detection

### 🎯 Objective  
Use **Support Vector Machines (SVM)** with both **linear** and **non-linear (RBF)** kernels to classify breast cancer diagnoses as **Malignant (1)** or **Benign (0)** using the Wisconsin Diagnostic Breast Cancer dataset.

---

## 📁 Dataset Summary

- **Source**: Breast cancer dataset (CSV)
- **Classes**:
  - 0 → Benign (357 samples)
  - 1 → Malignant (212 samples)
- **Total samples**: 569
- **Features**: 30 numeric attributes (e.g., radius, texture, area)
- **Preprocessing**:
  - Removed: `id`
  - Converted labels: M → 1, B → 0
  - Standardized features using `StandardScaler`
  - Applied PCA for 2D visualization
---

### ⚙️ Workflow

#### ✅ Preprocessing
- Removed unnecessary columns (`id`, `Unnamed: 32`)
- Converted `diagnosis` to binary (M → 1, B → 0)
- Standardized features using `StandardScaler`
- Split dataset: **80% Train / 20% Test**

---

### 🔍 SVM Training

#### 📌 Linear Kernel

- **Accuracy**: `95.61%`

```text
Confusion Matrix:
[[68  3]
 [ 2 41]]

Classification Report:
Precision (0): 0.97 | Recall (0): 0.96 | F1-score (0): 0.96  
Precision (1): 0.93 | Recall (1): 0.95 | F1-score (1): 0.94  
Overall Accuracy: 0.96
```

#### 📌 RBF Kernel
- **Accuracy**: `97.37%`

```text
Confusion Matrix:
[[70  1]
 [ 2 41]]

Classification Report:
Precision (0): 0.97 | Recall (0): 0.99 | F1-score (0): 0.98  
Precision (1): 0.98 | Recall (1): 0.95 | F1-score (1): 0.96  
Overall Accuracy: 0.97
```

---

### 🧪 Hyperparameter Tuning with GridSearchCV

#### 🔍 Parameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}
```

| Parameter | Description |
|-----------|-------------|
| `C`       | Regularization strength |
| `gamma`   | Kernel coefficient (influences decision boundary shape) |
| `kernel`  | Type of kernel to use in the algorithm |

### ✅ Best Parameters

```python
{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
```

---

### 📈 Model Evaluation (Best Estimator)

#### 🔹 Accuracy: **97.37%**

### 📊 Confusion Matrix

```
[[70  1]
 [ 2 41]]
```

### 🧾 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** (Benign)    | 0.97     | 0.99    | 0.98      | 71      |
| **1** (Malignant) | 0.98     | 0.95    | 0.96      | 43      |
| **Accuracy**      |          |         | **0.9737**| 114     |
| **Macro Avg**     | 0.97     | 0.97    | 0.97      | 114     |
| **Weighted Avg**  | 0.97     | 0.97    | 0.97      | 114     |

---

### 🧠 Interpretation

- **RBF Kernel**: Outperforms linear kernel by capturing complex decision boundaries.
- **Low false positives and negatives**: Excellent precision and recall for both classes.
- **Tuning effect**: GridSearchCV helped find optimal `C` and `gamma` for generalization.

---

### 📌 Why Hyperparameters Matter

| Hyperparameter | Role |
|----------------|------|
| `C`            | Controls trade-off between correct classification of training data and margin maximization |
| `gamma`        | Determines how much influence a single training example has |
| `kernel`       | Defines the function used to transform the data into a higher-dimensional space |

---

### ✅ Conclusion

Hyperparameter tuning using **GridSearchCV** improved the model accuracy to **97.37%**. The best model used:
- `C = 1`
- `gamma = 'scale'`
- `kernel = 'rbf'`

This shows the power of tuning and the effectiveness of RBF kernel in handling non-linear classification tasks in medical datasets.
