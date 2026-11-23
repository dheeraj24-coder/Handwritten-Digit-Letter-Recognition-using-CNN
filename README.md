# Handwritten-Digit-Letter-Recognition-using-CNN

# ✍️ Handwritten Digit & Letter Recognition using CNN

  **An AI-powered Handwritten Character Recognition System using Deep Learning and Computer Vision**

---

## 📌 About The Project

This project is a **Handwritten Digit and Letter Recognition System** developed using Convolutional Neural Networks (CNN), TensorFlow, and OpenCV.

The main objective is to build a single deep learning model that can recognize both handwritten digits (0–9) and uppercase English letters (A–Z).

Instead of building separate models for digits and letters, this project combines two different datasets, MNIST and EMNIST Letters, to train one CNN model capable of classifying 36 different characters.

The system also includes a **Live Webcam Recognition Feature**, which allows users to write a character on a white paper and see the model's prediction in real time.

## 🎯 Project Objectives

- 🔢 Recognize handwritten digits from 0 to 9.
- 🔤 Recognize handwritten uppercase letters from A to Z.
- 🧠 Build a single CNN model for all 36 character classes.
- 📊 Combine and preprocess MNIST and EMNIST datasets.
- 📈 Evaluate model performance using accuracy, loss, and confusion matrix.
- 📷 Implement real-time handwritten character recognition using a webcam.
- ⚡ Display the top 3 predicted characters with their confidence scores.

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🔢 Digit Recognition | Recognizes digits 0–9 |
| 🔤 Letter Recognition | Recognizes uppercase letters A–Z |
| 🧠 Single CNN Model | One model handles all 36 classes |
| 🔄 Data Augmentation | Rotation, zoom, and translation |
| 📊 Model Evaluation | Accuracy, loss, and confusion matrix |
| 📈 Training Visualization | Training vs validation graphs |
| 🎥 Live Webcam | Real-time character recognition |
| 🏆 Top-3 Predictions | Displays the three most likely predictions |
| 💾 Model Saving | Saves the trained model in `.keras` format |

## 🛠️ Technologies & Libraries Used

- **Python** – Main programming language.
- **TensorFlow** – Deep learning framework.
- **Keras** – Building and training the CNN model.
- **NumPy** – Numerical operations and data manipulation.
- **Matplotlib** – Data visualization and training graphs.
- **Seaborn** – Confusion matrix and class distribution visualization.
- **TensorFlow Datasets (TFDS)** – Loading the EMNIST Letters dataset.
- **OpenCV** – Image preprocessing and live webcam detection.
- **Scikit-learn** – Confusion matrix and model evaluation.

## 📂 Datasets Used

This project uses two publicly available handwritten character datasets.

### 1️⃣ MNIST Dataset

MNIST is a popular handwritten digit dataset.

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Color: Grayscale
- Classes: 10 (0–9)

**Loading the dataset:**

```python
(x_train_d, y_train_d), (x_test_d, y_test_d) = keras.datasets.mnist.load_data()
```

### 2️⃣ EMNIST Letters Dataset

EMNIST Letters contains handwritten English alphabet characters.

- Training Images: 124,800
- Testing Images: 20,800
- Image Size: 28 × 28 pixels
- Color: Grayscale
- Classes: 26 (A–Z)

**Loading the dataset:**

```python
(ds_train, ds_test), info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)
```

### 🔗 Dataset Combination

Both datasets are combined to create a single dataset containing 36 character classes.

| Dataset | Classes | Training Images | Testing Images |
|---|---:|---:|---:|
| MNIST | 10 | 60,000 | 10,000 |
| EMNIST Letters | 26 | 124,800 | 20,800 |
| **Total** | **36** | **184,800** | **30,800** |

**Final Classes:**

`0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`

## ⚙️ Project Workflow

The project follows these steps:

1. Import required libraries.
2. Load the MNIST digit dataset.
3. Load the EMNIST Letters dataset.
4. Visualize sample images.
5. Correct EMNIST image orientation.
6. Convert datasets into NumPy arrays.
7. Remap letter labels from 1–26 to 10–35.
8. Combine both datasets.
9. Normalize pixel values between 0 and 1.
10. Shuffle the training dataset.
11. Build the CNN architecture.
12. Compile and train the model.
13. Visualize training and validation performance.
14. Evaluate the model on test data.
15. Generate a confusion matrix.
16. Save the trained model.
17. Implement live webcam recognition.

## 🧹 Data Preprocessing

Data preprocessing is an important part of this project.

The following operations are performed:

- **Image Reshaping:** Convert images into the required CNN input shape `(28, 28, 1)`.
- **Image Orientation:** Transpose EMNIST images to correct their orientation.
- **Label Mapping:** Convert EMNIST labels from 1–26 into 10–35.
- **Normalization:** Scale pixel values from 0–255 to 0–1.
- **Dataset Merging:** Combine MNIST and EMNIST datasets.
- **Shuffling:** Randomly shuffle training samples to mix digits and letters.

## 🧠 CNN Model Architecture

A Convolutional Neural Network (CNN) is used to recognize handwritten characters.

### Model Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 28 × 28 × 1 | Input image |
| Data Augmentation | Rotation, Zoom, Translation | Improve generalization |
| Conv2D | 32 filters, 3×3 | Extract image features |
| MaxPooling2D | 2×2 | Reduce dimensions |
| Conv2D | 64 filters, 3×3 | Extract deeper features |
| MaxPooling2D | 2×2 | Reduce dimensions |
| Flatten | — | Convert features into 1D |
| Dropout | 0.5 | Reduce overfitting |
| Dense | 256 neurons | Learn complex patterns |
| Output | 36 neurons, Softmax | Predict character class |

### CNN Architecture Flow

```text
Input Image (28x28x1)
        |
        v
Data Augmentation
        |
        v
Conv2D (32 Filters)
        |
        v
MaxPooling2D
        |
        v
Conv2D (64 Filters)
        |
        v
MaxPooling2D
        |
        v
Flatten
        |
        v
Dropout (0.5)
        |
        v
Dense (256)
        |
        v
Softmax (36 Classes)
        |
        v
Predicted Character
```

## ⚡ Model Training

The CNN model is compiled and trained using the following configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Epochs | 6 |
| Batch Size | 128 |
| Validation Split | 10% |
| Output Classes | 36 |

**Model Compilation:**

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**Model Training:**

```python
history = model.fit(
    x_train,
    y_train,
    epochs=6,
    batch_size=128,
    validation_split=0.1
)
```

### 🛑 Callbacks

EarlyStopping and ModelCheckpoint are also defined in the notebook to support training optimization and saving the best model.

- **EarlyStopping:** Stops training when validation loss stops improving.
- **ModelCheckpoint:** Saves the model with the best validation loss.

*Note: To activate these callbacks, pass `callbacks=[early_stopping, checkpoint]` inside `model.fit()`.*

## 📊 Model Evaluation

After training, the model is evaluated using the test dataset.

The evaluation includes:

- Overall Test Accuracy
- Digit Recognition Accuracy
- Letter Recognition Accuracy
- Training and Validation Accuracy
- Training and Validation Loss
- Confusion Matrix
- Top 5 Most Confused Character Pairs

**Prediction Example:**

```python
y_prob = model.predict(x_test, verbose=0)

y_pred = np.argmax(y_prob, axis=1)
```

**Accuracy Calculation:**

```python
overall = (y_pred == y_test).mean()

is_digit = y_test < 10

print("Overall Accuracy:", overall * 100)
print("Digit Accuracy:", 
      (y_pred[is_digit] == y_test[is_digit]).mean() * 100)

print("Letter Accuracy:", 
      (y_pred[~is_digit] == y_test[~is_digit]).mean() * 100)
```

### 📈 Performance Visualization

The project includes:

- Training vs Validation Accuracy Graph
- Training vs Validation Loss Graph
- Confusion Matrix Heatmap
- Random Prediction Visualization

**Note:** Actual accuracy depends on training, model configuration, and dataset processing. Add your final measured test accuracy here after running the notebook.

## 📷 Live Webcam Recognition

One of the most interesting features of this project is real-time handwritten character recognition using OpenCV.

The webcam captures the character written inside a predefined region.

### How It Works

1. Open the webcam.
2. Display a green recognition box.
3. Write a digit or uppercase letter on white paper using a thick black marker.
4. Place the character inside the green box.
5. Capture the Region of Interest (ROI).
6. Convert the image into grayscale.
7. Apply Gaussian Blur.
8. Apply thresholding.
9. Remove small noise using morphological operations.
10. Crop and center the character.
11. Resize it to 20 × 20 pixels.
12. Add a 4-pixel black border to create a 28 × 28 image.
13. Pass the image to the trained CNN model.
14. Display the top 3 predictions with confidence percentages.

### 🎮 Webcam Controls

| Key | Action |
|---|---|
| `q` | Quit the webcam window |

### 💡 Webcam Tips

- Use good lighting.
- Write with a thick black marker.
- Use white paper.
- Write only one character at a time.
- Prefer uppercase English letters.
- Keep the character inside the green box.

## 📁 Project Structure

```text
Handwritten-Recognition-CNN/
│
├── Handwritten_Recognition_CNN_P1.ipynb
│
├── digit_letter_model.keras
│
├── best_char_model.keras
│
├── README.md
│
└── requirements.txt
```

*The model files are generated after running the notebook. `best_char_model.keras` is only generated when ModelCheckpoint is passed to model.fit().*

## 💻 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Handwritten-Recognition-CNN.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Handwritten-Recognition-CNN
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install tensorflow tensorflow-datasets opencv-python matplotlib numpy scikit-learn seaborn jupyter
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 6: Run the Notebook

Open:

`Handwritten_Recognition_CNN_P1.ipynb`

Run all cells sequentially, from the beginning to the end.

**Important:** The EMNIST dataset downloads automatically the first time it is loaded. An internet connection is required.

## 💾 Saving and Loading the Model

After training, the model is saved in Keras format.

**Save Model:**

```python
model.save("digit_letter_model.keras")
```

**Load Model:**

```python
model = keras.models.load_model("digit_letter_model.keras")
```

The saved model can be loaded later without training again.

## ⚠️ Limitations

Although the model recognizes 36 different characters, it has some limitations:

- It recognizes only one character at a time.
- It does not recognize complete words or sentences.
- Similar-looking characters can create confusion.
- Handwriting styles can affect predictions.
- Poor lighting can affect webcam performance.
- Recognition accuracy may vary depending on image quality.

### 🔍 Common Character Confusion

Some characters have similar visual appearances:

| Character Pair | Reason |
|---|---|
| 0 and O | Similar shape |
| 1 and I | Similar appearance |
| 5 and S | Similar curves |

## 🔮 Future Improvements

The project can be extended with several advanced features:

- 📝 Full handwritten word recognition.
- 📄 Handwritten sentence recognition.
- 🌐 Web-based character recognition application.
- 📱 Mobile application integration.
- ⚡ Improved model accuracy and generalization.
- 🎯 Better handwriting detection and segmentation.
- 🤖 Advanced OCR using CRNN and attention-based models.
- 🌍 Support for additional languages and scripts.

## 🎓 Learning Outcomes

Through this project, I explored and implemented:

- Understanding of CNN architecture.
- Working with MNIST and EMNIST datasets.
- Image preprocessing and normalization.
- Data augmentation techniques.
- Multi-class image classification.
- Model training and validation.
- Performance evaluation.
- Confusion matrix analysis.
- Model saving and loading.
- Real-time computer vision using OpenCV.
- Integration of Deep Learning with webcam applications.

## 👨‍💻 Developer

**Dheeraj Chouhan**

Aspiring AI/ML Engineer | Python | Deep Learning | Computer Vision

- 🐙 GitHub: https://github.com/dheeraj24-coder
- 💼 LinkedIn: https://www.linkedin.com/in/dheeraj-chouhan-7a6996371/

---

  **⭐ If you find this project interesting, don't forget to star the repository!**

  Made with ❤️ using Python, TensorFlow, CNN & OpenCV.
