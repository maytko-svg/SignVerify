# SignVerify - Signature Forgery Detection

A web app for detecting forged handwritten signatures using ResNet50 deep learning.

**🚀 [Live Demo](https://signverify-6sqztqzv2cbrjxqxw93h8e.streamlit.app/)**

## Features
- Upload reference and query signatures
- ResNet50 deep learning model
- Real-time forgery detection with confidence scores
- Grad-CAM attention heatmaps

## Project Files
- `app-yellowblack.py` - Streamlit web application
- `resnet50_signature.ipynb` - Model training & evaluation
- `requirements.txt` - Dependencies

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app-yellowblack.py
```

## Model & Dataset
- **Architecture:** ResNet50 (transfer learning)
- **Training Data:** Combined CEDAR + Real-Fake Signature Datasets
  - [CEDAR Signature Dataset] (https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
  - [Real-Fake Signature Datasets](https://www.kaggle.com/datasets/emrahaydemr/realfake-signature-datasets)
- **Model Storage:** Google Drive (auto-downloaded by Streamlit app)

## References
[CEDAR] Cha, S. H., Yoon, H., & Tappert, C. C. (2005). Enhancing Binary Feature...
