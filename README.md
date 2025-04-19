# NeuroGPT
**Neuro-GPT: Towards a Foundation Model for EEG**  
**Published at IEEE ISBI 2024**

We propose **Neuro-GPT**, a foundation model consisting of an EEG encoder and a GPT model. The model is pre-trained on a large-scale dataset using a self-supervised task that learns to reconstruct masked EEG segments. To validate its capability in low-data regimes, we fine-tune Neuro-GPT on a Motor Imagery Classification task involving only 9 subjects.

Our experiments demonstrate that applying a foundation model significantly improves classification performance compared to training from scratch.

---

## 🔗 Pre-trained Foundation Model  
> [Download here](#) <!-- Replace with actual link if available -->

---

## 🧠 Neuro-GPT Pipeline  

```text
EEG Signal → EEG Encoder → Masked Modeling → Pre-training → Fine-tuning (MI Task) → Classification Output
```

---

## 🛠 Installation  

```bash
git clone git@github.com:wenhui0206/NeuroGPT.git
cd NeuroGPT
pip install -r conda_requirements.txt
cd scripts
./train.sh
```

---

## 📦 Requirements  

```bash
pip install -r conda_requirements.txt
```

---

## 📚 Datasets  

- [TUH EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)  
- [BCI Competition IV 2a Dataset](http://www.bbci.de/competition/iv/)

---

## 🙏 Acknowledgments  

This project is developed based on the following open-source repositories:

- [Self-supervised learning of brain dynamics from broad neuroimaging data](https://github.com/neurodataset/self-supervised-eeg)
- [EEG-Conformer](https://github.com/ShawnBIT/EEG-Conformer)
