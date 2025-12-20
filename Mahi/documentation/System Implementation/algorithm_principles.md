# Algorithm Principles for Beginners

This guide explains the three machine learning algorithms used in our project: **Convolutional Neural Networks (CNN)**, **Multi-Layer Perceptrons (MLP)**, and **Random Forests**. We use these "brains" to teach the computer how to recognize digits.

---

## 1. Convolutional Neural Networks (CNN)

Imagine looking at a picture through a small square hole (a window) that you slide across the entire image. This is essentially what a CNN does!

### Key Concepts

*   **Convolution (The "Slide and Scan")**:
    *   Think of a **Filter** (or kernel) as a small pattern detector, like a flashlight looking for specific shapes—curves, edges, or corners.
    *   The CNN slides this filter over the image pixel by pixel. When the filter matches a part of the image, it "lights up" (produces a high value).
    *   **Why?** This helps the computer focus on *features* (like the loop in a '6' or the cross in a '4') rather than just individual dots.

*   **Pooling (The "Summarizer")**:
    *   After finding features, we often have too much information. **Max Pooling** looks at a small patch of results and just keeps the biggest number (the strongest match).
    *   **Why?** It makes the image smaller and easier to process while keeping the most important details. It also helps the model recognize the object even if it's shifted slightly.

*   **ReLU (The "Filter Switch")**:
    *   Stands for **Re**ctified **L**inear **U**nit. It's a fancy name for a simple rule: "If the value is negative, make it zero."
    *   **Why?** It adds non-linearity, allowing the model to learn complex patterns instead of just straight lines.

### In Our Project
We use CNNs (`FinalCNN`, `SimpleDigitCNN`, `SimpleCNN`) because they are heavily specialized for image recognition. They are excellent at understanding the spatial structure of our digit images.

---

## 2. Multi-Layer Perceptron (MLP)

An MLP is the classic "Neural Network." If a CNN is like scanning an image with a window, an MLP is like looking at all the pixels at once as a long list.

### Key Concepts

*   **Flattening**:
    *   Our images are 2D grids (e.g., 28 rows x 28 columns). An MLP can't read a grid; it needs a list.
    *   **Flattening** takes all the rows and lines them up into one long string of numbers (e.g., 28x28 = 784 numbers).

*   **Neurons and Layers**:
    *   **Input Layer**: Receives the raw pixel data (the list of 784 numbers).
    *   **Hidden Layers**: The "thinking" middle parts. Each neuron in a layer connects to every neuron in the next layer. They do math to combine the inputs.
    *   **Output Layer**: The final decision. For digits, we have 10 output neurons (one for each digit 0-9). The highest value wins!

*   **Weights and Biases**:
    *   Think of these as "strength knobs." During training, the model tweaks these knobs to get better answers.

### In Our Project
We use MLPs (`MNIST_MLP`, `SimpleMLP`) to show a different approach. While powerful, they can struggle if objects move around in the image because they don't understand "neighbors" as well as CNNs do.

---

## 3. Random Forest

Unlike the "brain-like" neural networks above, a Random Forest is more like a council of voters.

### Key Concepts

*   **Decision Tree**:
    *   Imagine a game of "20 Questions." Is the top pixel dark? (Yes/No). Is the middle pixel bright? (Yes/No).
    *   A Decision Tree asks a chain of these Yes/No questions to arrive at an answer.

*   **The "Forest" (Ensemble)**:
    *   A single tree might make mistakes. So, we create a whole **forest** of many different trees.
    *   Each tree gets a slightly different random subset of the data to learn from.
    *   **Voting**: When we show the forest an image, every single tree makes a guess. The digit that gets the most votes is the final winner.

### In Our Project
We use Random Forest (`RandomForestClassifier`) as a strong baseline. It doesn't need as much data preprocessing or "training time" on a GPU as neural networks, and it's often surprisingly accurate!
