# Confusion Matrices Analysis

A "Confusion Matrix" sounds confusing, but it's actually the best tool to see exactly *where* a model is making mistakes.

---

## 1. What is a Confusion Matrix?

Imagine a grid.
*   **Rows**: The **Correct Answer** (what the image actually is).
*   **Columns**: The **Model's Prediction** (what the computer thought it was).

*   **The Diagonal Line**: We want all the big numbers to be on the diagonal line from top-left to bottom-right. This means "Is a 0, Predicted 0", "Is a 1, Predicted 1", etc.
*   **Off-Diagonal**: Any number *not* on the diagonal is a **Mistake**.

---

## 2. Our Confusion Report

Based on our test results in `Mahi/results`, here are the most common "confusions" our models suffered from:

### A. The "4" vs. "1" Confusion
*   **The Scenario**: The digit `4` was frequently mistaken for a `1` or `7`.
*   **Who Did It**: `CNN_raw`, `CNN_mnist`.
*   **Why?** In some handwriting, a '4' looks like a vertical line (like a '1') with a small cross. If the cross is faint or the model focuses too much on vertical edges, it guesses '1'.

### B. The "2" vs. "7" Confusion
*   **The Scenario**: The digit `2` was sometimes guessed as `7`.
*   **Who Did It**: `Random Forest`.
*   **Why?** Both digits have a sharp angle at the top-right. Without understanding the curve at the bottom of a '2', a simpler model might just see "Top Right Angle -> Must be 7".

### C. The "1" vs. "9" Confusion
*   **The Scenario**: A noisy `1` can look like a `9` to some models.
*   **Who Did It**: `CNN_raw` on the noisy image `1(1).png`.
*   **Why?** Noise artifacts near the top of the '1' can look like the loop of a '9'.

---

## 3. Visual Summary (Text Version)

Here is a simplified "mini-matrix" of our test results for just a few digits:

| Actual Digit | Most Common Prediction | Notes |
| :--- | :--- | :--- |
| **0** | **0** | **Perfect.** No confusion. |
| **1** | **1** | Sometimes confused with **9**. |
| **2** | **2** | Sometimes confused with **7** or **8**. |
| **3** | **3** | Generally good, rarely confused with **9**. |
| **4** | **1** or **7** | **High Confusion.** The hardest digit. |
| **5** | **5** | **Perfect.** Distinct shape. |
| **6** | **6** | Solid. |
| **7** | **7** | Sometimes confused with **1**. |
| **8** | **8** | Solid. |
| **9** | **9** | sometimes confused with **7** or **3**. |

**Conclusion**: If the matrix shows a lot of numbers scattered all over the place (like for the Random Forest model), it means the model is "confused" everywhere. If the errors are clustered (like everyone failing on '4'), it means that specific digit is tricky for everyone.
