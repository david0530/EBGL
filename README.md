# EEG-Transformer

**EEG-Transformer: A Transformer-Based Deep Learning Framework for EEG Analysis**

This project implements a Transformer-based deep learning model for analyzing EEG signals. It features modular layer definitions, flexible model construction, and GPU-optimized utilities for training and evaluation.

---

## 📁 Project Structure

- `layers.py` – Defines core neural network components such as attention blocks and embedding layers.
- `models.py` – Builds full model architectures using the custom layers.
- `utill_gpu.py` – Includes GPU training utilities, seed setup, accuracy and performance tracking functions.

---

## 🚀 Features

- Transformer-based architecture tailored for EEG data
- Flexible model definition for both classification and regression tasks
- GPU-accelerated training pipeline
- Easily extensible for additional tasks or datasets

---

## 🛠 Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

Install dependencies:

```bash
pip install -r conda_requirements.txt
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

---

## 🧪 Usage

To train the model:

```bash
cd scripts
./train.sh
```

You can also manually run:

```python
python train.py --config=configs/config.yaml
```

Modify `train.py` and configuration files as needed for your custom dataset.

---

## 📚 Dataset

This framework is designed to work with EEG datasets. You may integrate datasets such as:

- TUH EEG Corpus
- BCI Competition datasets
- CHB-MIT Scalp EEG

---

## 🙏 Acknowledgments

This work is inspired by state-of-the-art transformer architectures and EEG research.

---

## 📄 License

This project is licensed under the MIT License.
