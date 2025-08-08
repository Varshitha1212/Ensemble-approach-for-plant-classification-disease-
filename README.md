[README (1).md](https://github.com/user-attachments/files/21692633/README.1.md)
# 🌱 Ensemble Approach for Plant Disease Classification

An advanced deep learning project that classifies plant diseases using an **ensemble** of multiple pre-trained CNN architectures — **InceptionV3**, **ResNet50**, **VGG16**, and **VGG19** — fine-tuned on plant leaf images.

---

## 📌 Features
- Combines predictions from **four powerful CNN architectures** for higher accuracy.
- Uses **transfer learning** with pre-trained ImageNet weights.
- Designed for **plant disease detection** from leaf images.
- Modular and easy-to-extend architecture.
- Can be integrated into a web app or mobile app.

---

## 🛠 Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**
- **scikit-learn**
- **OpenCV**

---

## 📂 Project Structure
```
📦 Ensemble-approach-for-plant-disease
 ┣ 📜 inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
 ┣ 📜 resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
 ┣ 📜 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
 ┣ 📜 vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
 ┣ 📜 train.py           # Script to train the ensemble model
 ┣ 📜 predict.py         # Script to run predictions on new images
 ┣ 📜 requirements.txt   # Python dependencies
 ┗ 📜 README.md          # Project documentation
```

---

## 🚀 How It Works
1. Each model (**InceptionV3**, **ResNet50**, **VGG16**, **VGG19**) is loaded with pre-trained weights.
2. Input plant leaf image is preprocessed and fed into all models.
3. Predictions from each model are **averaged (or weighted)** to get the final classification.
4. Output is the predicted **disease name** or **healthy leaf**.

---

## 🖥 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Ensemble-approach-for-plant-disease.git
cd Ensemble-approach-for-plant-disease
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Prediction
```bash
python predict.py --image path_to_image.jpg
```

---

## 📊 Example Output
```
Input: tomato_leaf.jpg
Predicted: Tomato Leaf Mold
Confidence: 97.4%
```

*(Add sample images or screenshots here)*

---

## 📜 License
This project is open-source and available under the MIT License.

---

## 👩‍💻 Author
**Varshitha Malladi**  
📌 GitHub: [Your GitHub Profile](https://github.com/yourusername)

