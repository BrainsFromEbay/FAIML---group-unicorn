# Learning Curves Analysis

"Learning Curves" are graphs that tell us how the model improves over time while it's "at school" (training). Since we didn't save the physical image files of the charts, this report explains the trends we observed in the training logs.

---

## 1. What are Learning Curves?

There are two main lines we not mally watch:

### A. The Loss Curve (The "Mistake Counter")
*   **What it is**: Measures how wrong the model is.
*   **Goal**: We want this line to go **DOWN** to zero.
*   **Our Observation**:
    *   **CNNs**: Dropped very fast! In just 1-2 epochs, the loss went from high numbers to nearly 0.0. This means CNNs learn extremely quickly.
    *   **MLPs**: Dropped slower and sometimes got "stuck" (fluctuated). This shows they had a harder time figuring out the patterns.

### B. The Accuracy Curve (The "Grade")
*   **What it is**: Measures what percentage of questions the model got right.
*   **Goal**: We want this line to go **UP** to 100%.
*   **Our Observation**:
    *   **CNNs**: Shot up to 90%+ accuracy almost immediately.
    *   **Random Forest**: Doesn't have a "curve" like neural networks because it doesn't learn in epochs; it just learns everything at once.

---

## 2. Common Patterns We Saw

### The "Good Student" (CNNs)
*   **Pattern**: Loss goes down steadily, Validation Accuracy goes up steadily.
*   **Meaning**: The model is learning well and generalizing to new data.
*   **Example**: `CNN_raw` followed this path perfectly, reaching 85% validation accuracy.

### The "Memorizer" (Overfitting)
*   **Pattern**: Training Accuracy hits 100% (perfect score), but Validation Accuracy stops improving or even drops.
*   **Meaning**: The model is just memorizing the answers instead of learning the rules. It fails when it sees new questions.
*   **Example**: `MLP_pickle` showed signs of this. It would get very high scores during training but then struggle on the `custom_test` images (only 70% accuracy).

### The "Confused Student" (Underfitting)
*   **Pattern**: Both Training and Validation accuracy stay low. The model just can't figure it out.
*   **Example**: `Random Forest` (45% accuracy). The line would stay flat and low, indicating the model structure just wasn't smart enough for this complex image task.

---

## 3. Summary of Training logs

Based on the logs printed during our attempts:
*   **Fastest Learner**: `CNN_mnist` (Learns in minutes because it uses a simple, strong structure).
*   **Slowest Learner**: `MLP_pickle` (Needs more epochs to adjust its many weights).
*   **Most Stable**: `CNN_raw` (Consistent improvement without wild jumps).
