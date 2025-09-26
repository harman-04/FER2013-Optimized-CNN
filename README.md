# FER2013-Optimized-CNN

### Optimized CNN Pipeline for Facial Expression Recognition (FER) on the Challenging FER2013 Dataset

This repository contains the complete code for a deep learning pipeline designed to perform Facial Expression Recognition (FER) with high accuracy on the FER2013 dataset. The project directly addresses the primary challenges of this benchmark: low image resolution (48x48 pixels) and severe class imbalance.

By implementing advanced data preprocessing, a robust Convolutional Neural Network (CNN) architecture, and sophisticated training strategies, this model achieves a significant improvement over common baselines.

---

### 🌟 Key Features

-   **Optimized Data Processing:** A dedicated class for loading and preparing the FER2013 dataset, including parsing pixel data and applying Contrast Limited Adaptive Histogram Equalization (CLAHE) for feature enhancement.
-   **Smart Class Balancing:** A custom, conservative oversampling strategy and dynamic class weighting are used to mitigate the dataset's severe class imbalance, particularly for the 'disgust' emotion.
-   **Robust CNN Architecture:** A VGG-like deep CNN model with Batch Normalization and Dropout layers to prevent overfitting and ensure stable training.
-   **Advanced Training Callbacks:** Implements `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to dynamically adjust the learning rate, leading to faster convergence and better performance.
-   **Comprehensive Evaluation:** Provides a detailed analysis of model performance, including a full classification report, visual confusion matrices (both raw and normalized), and an in-depth breakdown of top error patterns.
-   **Reproducible Pipeline:** The entire pipeline is designed to be easily runnable, with options for full retraining or quick evaluation of a pre-trained model.

---

**Note:** The `FER2_013.ipynb` script assumes the `fer2013.csv` file is located in your Google Drive at the specified path (`/content/drive/MyDrive/fer2013.csv`).

---

### 🚀 Usage

This script is designed to be run in a Google Colab environment due to its integration with Google Drive and the computational demands of training.

#### 1. Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harman-04/FER2013-Optimized-CNN.git](https://github.com/harman-04/FER2013-Optimized-CNN.git)
    cd FER2013-Optimized-CNN
    ```
    
2.   **Dowload Dataset:** https://drive.google.com/file/d/11MPfZwi1TheA-RMV0a4mgVUG1VfpScLO/view?usp=drive_link
3.   **Upload Dataset:** Place the `fer2013.csv` file in your Google Drive.
4.  **Open in Google Colab:** Open the `FER2_013.ipynb` file in Google Colab.
5.  **Install Dependencies:** Run the following command in a code cell to install the necessary libraries:
    ```bash
    !pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imblearn opencv-python
    ```
6.  **Run the Script:** Execute the script. It will prompt you to choose an operational mode.

#### 2. Running the Pipeline

The script provides three modes of operation:

-   **Full training pipeline (`1`):** Trains the model from scratch. This can take a considerable amount of time depending on your hardware (GPU).
-   **Quick evaluation only (`2`):** Loads a pre-trained model and performs a full evaluation on the test set. Useful for verifying the model's performance.
-   **Performance recovery (`3`):** The recommended mode. It will retrain the model but is designed to handle issues like performance degradation over time, restarting training from a new, stable point.

---

### 📈 Results & Analysis

The pipeline is engineered to achieve a test accuracy of approximately **64.78%**, representing a significant improvement over the baseline accuracy for this dataset.

#### Key Performance Metrics
The model shows particularly strong performance on the most frequent classes. A key achievement is the substantial recall for the under-represented `disgust` class, demonstrating the effectiveness of the class-weighting strategy.

| Emotion | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Angry** | 0.6121 | 0.6409 | 0.6262 |
| **Disgust** | 0.8123 | 0.5854 | 0.6811 |
| **Fear** | 0.5097 | 0.4439 | 0.4746 |
| **Happy** | 0.7788 | 0.8503 | 0.8130 |
| **Sad** | 0.5976 | 0.5877 | 0.5926 |
| **Surprise** | 0.8173 | 0.8133 | 0.8153 |
| **Neutral** | 0.6011 | 0.6344 | 0.6173 |

#### Normalized Confusion Matrix

The normalized confusion matrix below provides a clear visual representation of where the model excels and where it struggles. The dark blue diagonal indicates high recall for a given class. The matrix reveals common confusions, such as between `Fear` and `Sad` or `Sad` and `Neutral`, which are often plausible even for human observers.

!<img width="831" height="343" alt="image" src="https://github.com/user-attachments/assets/4516a2d9-0c6d-428e-aaed-087b404f88db" />


---

### ✍️ Conclusion

The FER2013-Optimized-CNN project provides a comprehensive and effective solution for the challenging task of Facial Expression Recognition. By systematically addressing low resolution, class imbalance, and model stability, this pipeline serves as a strong foundation for further research and development in affective computing.

