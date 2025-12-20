# Comparison of Models

This report compares how well our different "brains" (models) performed when tested on a set of 20 challenging digit images.

---

## 1. The Scoreboard (Accuracy)

We tested every model on 20 images. Here is the final ranking:

| Rank | Model Name | Source Code | Accuracy | Grade |
| :--: | :--- | :--- | :---: | :--- |
| 🥇 **1st** | **CNN (Raw Images)** | `CNN_raw.py` | **85%** | **A** |
| 🥈 **2nd** | **CNN (Pickle)** | `CNN_pickle.py` | **80%** | **A-** |
| 🥈 **2nd** | **CNN (MNIST)** | `CNN_mnist.py` | **80%** | **A-** |
| 4th | MLP (MNIST) | `MLP_mnist.py` | 75% | B |
| 5th | MLP (Pickle) | `MLP_pickle.py` | 70% | C+ |
| 6th | Random Forest | `random_forrest.py` | 45% | F |

### Key Takeaways
1.  **CNNs are Superior**: All Top 3 models are Convolutional Neural Networks. They just "see" images better than other methods.
2.  **Random Forest Struggled**: It scored less than 50%, meaning it was guessing wrong more often than right. It is not suitable for this specific complex image task.
3.  **Data Source Matters**: The model trained on **Raw Images** slightly beat the ones trained on Pickle or pure MNIST data.

---

## 2. Detailed Match Analysis

We looked at specific "problem images" to see where models failed.

### The "Digit 4" Problem
The digit `4` was the hardest test.
*   **Result**: Almost EVERY model failed to recognize `4.png`.
*   **The Exception**: Only **MLP (Pickle)** correctly identified it.
*   **Lesson**: Sometimes a generally "weaker" model (like MLP) learns something unique that the "stronger" models (CNNs) miss. This is why scientists sometimes use "Ensembles" (combining multiple models).

### The "Noisy" Images
half of our test images had "noise" (preprocessing artifacts).
*   **Observation**: Surprisingly, models trained on **Pickled Data** handled these noisy images perfectly.
*   **Why?** The pickled training data likely had similar noise patterns, so the model learned to ignore them. The "Raw" model, which didn't see that specific noise during training, sometimes got confused.

---

## 3. Recommendation

If we had to pick just one model to use in a real app:
> **Use the `CNN_raw` model.**

However, for a super-smart system:
> **Combine `CNN_raw` + `CNN_pickle` + `MLP_pickle`.**
> *   `CNN_raw` gives general accuracy.
> *   `CNN_pickle` handles noisy images.
> *   `MLP_pickle` catches the difficult '4's.
