# ✍️ Character Recognition App

A real-time handwritten character recognition application built with **Streamlit**, **TensorFlow/Keras**, and **OpenCV**. Users can draw digits (0-9), uppercase letters (A-Z), and lowercase letters (a-z) on an interactive canvas and get instant predictions from a deep learning model.

## 🚀 Features

-   **Interactive Drawing Canvas**: Smooth drawing experience with adjustable brush and eraser sizes.
-   **Real-time Prediction**: Deep learning model classifies characters across 62 different classes.
-   **Confidence Scoring**: Displays the model's confidence percentage for each prediction.
-   **Modern UI**: Clean, responsive interface with a side-by-side layout for drawing and results.

## 🧠 Model Details

The underlying model is a Convolutional Neural Network (CNN) trained on the EMNIST (Extended MNIST) ByClass dataset.

### Dataset
-   **Source**: EMNIST ByClass (`emnist-byclass-train.csv`)
-   **Classes**: 62 (0-9, A-Z, a-z)
-   **Input Size**: 28x28 grayscale images.
-   **Preprocessing**: Pixel values normalized to `[0, 1]` and reshaped to `(28, 28, 1)`.

### Architecture
The model utilizes a sequential architecture designed for feature extraction from stroke patterns:

| Layer | Type | Details |
| :--- | :--- | :--- |
| 1 | **Input** | (28, 28, 1) |
| 2 | **Conv2D** | 32 filters, (3,3) kernel, ReLU |
| 3 | **MaxPooling2D** | (2,2) pool size |
| 4 | **Conv2D** | 64 filters, (3,3) kernel, ReLU |
| 5 | **MaxPooling2D** | (2,2) pool size |
| 6 | **Conv2D** | 128 filters, (3,3) kernel, ReLU |
| 7 | **MaxPooling2D** | (2,2) pool size |
| 8 | **Flatten** | - |
| 9 | **Dense** | 128 units, ReLU |
| 10 | **Dropout** | 0.4 rate |
| 11 | **Dense (Output)** | 62 units, Softmax |

### Training Performance
-   **Optimizer**: Adam
-   **Loss Function**: Sparse Categorical Crossentropy
-   **Epochs**: 10
-   **Final Accuracy**: ~83%
-   **Note**: The dataset is known to be imbalanced, which impacts the classification of similar-looking characters (e.g., 'O' vs 'o' vs '0').

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd "streamlit hand recognition app"
    ```

2.  **Install dependencies**:
    ```bash
    pip install streamlit streamlit-drawable-canvas tensorflow numpy pillow
    ```

3.  **Place the model**:
    Ensure your trained model file is named `model2.keras` and placed in the root directory.

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

1.  Select the **Brush** tool from the sidebar.
2.  Draw a single character in the center of the black canvas.
3.  Click the **Predict** button to see the results.

## 🛠️ Technologies Used

-   **Streamlit**: Frontend UI framework.
-   **TensorFlow/Keras**: Model building and inference.
-   **Streamlit Drawable Canvas**: Drawing interface.
-   **Pillow (PIL)**: Image processing and resizing.