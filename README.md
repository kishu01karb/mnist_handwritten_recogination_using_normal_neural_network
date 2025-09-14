# ✍️ MNIST Handwritten Digit Recognition using Neural Network

This project focuses on recognizing **handwritten digits (0-9)** using a **Neural Network** trained on the **MNIST dataset**.  
It demonstrates the fundamentals of **Deep Learning**, including building, training, and evaluating a neural network from scratch using Python.

---

## 📂 Project Structure
```
mnist_handwritten_recognition_using_normal_neural_network.ipynb  # Main Jupyter Notebook
README.md                                                         # Project documentation
```

---

## 🚀 Features
- Load and preprocess the MNIST dataset of handwritten digits.
- Build a simple **feedforward neural network** without using complex architectures like CNNs.
- Train the network to classify digits (0-9).
- Evaluate model performance using accuracy and confusion matrix.
- Visualize predictions with sample test images.

---

## 🛠 Tech Stack
- **Programming Language:** Python 🐍
- **Libraries Used:**
  - `tensorflow` / `keras` - Building and training the neural network
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Visualizing evaluation metrics
  - `scikit-learn` - Confusion matrix and evaluation tools
  - `jupyter` - Notebook environment

---

## 📊 Workflow
1. **Import Libraries** – Load essential Python libraries for deep learning.
2. **Load Dataset** – Import the MNIST dataset of handwritten digits.
3. **Data Preprocessing**  
   - Normalize pixel values for faster convergence.
   - Split dataset into training and testing sets.
4. **Build the Neural Network**
   - Input Layer: 784 neurons (28x28 image pixels flattened).
   - Hidden Layers: Dense layers with activation functions.
   - Output Layer: 10 neurons (for digits 0-9) with softmax activation.
5. **Compile the Model**
   - Optimizer: Adam or SGD
   - Loss Function: Categorical Crossentropy
   - Metrics: Accuracy
6. **Train the Model**
   - Fit the model using the training dataset.
7. **Evaluate the Model**
   - Measure accuracy on the test dataset.
   - Generate a confusion matrix.
8. **Make Predictions**
   - Predict digits for unseen test images and visualize the results.

---

## 📥 Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/mnist-handwritten-digit-recognition.git
cd mnist-handwritten-digit-recognition
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage
To run the project:
```bash
jupyter notebook
```
Then open `mnist_handwritten_recognition_using_normal_neural_network.ipynb` and execute the cells step-by-step.

---

## 📈 Results
- The trained model achieves high accuracy on the MNIST dataset.
- Visualizations show how the model classifies digits.

Example:
| **Input Image** | **Predicted Digit** |
|----------------|---------------------|
| ✏️ (28x28 digit "7") | 7 |
| ✏️ (28x28 digit "3") | 3 |

Confusion matrix and sample predictions are also displayed to analyze performance.

---

## 📜 License
This project is licensed under the MIT License.  
Feel free to use and modify for learning and experimentation.

---

## 👤 Author
- **Krishna Karbhari**
- GitHub: [kishu01karb](https://github.com/kishu01karb)

---

## 🌟 Acknowledgements
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- Inspiration from classic deep learning tutorials on digit recognition.
