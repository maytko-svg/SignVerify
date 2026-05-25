import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.nn.functional import cosine_similarity
from PIL import Image
import io
import os
import gdown

# Page config 
st.set_page_config(
    page_title = "SignVerify",
    page_icon = "✍️",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Session state 
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ref_image_stored' not in st.session_state:
    st.session_state.ref_image_stored = None
if 'query_image_stored' not in st.session_state:
    st.session_state.query_image_stored = None

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    :root {
        --bg: #111111;
        --card: #1e1e1e;
        --card-hover: #252525;
        --gold: #c9a96e;
        --gold-light: #d4b483;
        --gold-glow: rgba(201, 169, 110, 0.25);
        --text: #ffffff;
        --text-muted: #8a8a8a;
        --border: rgba(201, 169, 110, 0.30);
        --danger: #cf6679;
        --danger-glow: rgba(207, 102, 121, 0.25);
    }

    * { font-family: 'DM Sans', sans-serif !important; }

    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Primary action button */
    [data-testid="stButton"] { text-align: center !important; }
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #c9a96e 0%, #b8924a 100%) !important;
        color: #111111 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        padding: 0.85rem 3.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px var(--gold-glow) !important;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px var(--gold-glow) !important;
        background: linear-gradient(135deg, #d4b483 0%, #c9a96e 100%) !important;
    }
    [data-testid="stButton"] button:disabled {
        background: #2a2a2a !important;
        color: #444444 !important;
        box-shadow: none !important;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .metric-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--gold);
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: var(--text);
        font-family: 'DM Serif Display', serif !important;
    }
    .metric-bar-bg {
        background: #2c2c2c;
        border-radius: 4px;
        height: 5px;
        margin-top: 0.75rem;
    }
    .metric-bar-fill {
        height: 5px;
        border-radius: 4px;
    }

    /* Verdict */
    .verdict-genuine {
        font-size: 3.4rem;
        font-weight: 700;
        color: var(--gold);
        text-align: center;
        font-family: 'DM Serif Display', serif !important;
        text-shadow: 0 0 30px var(--gold-glow);
        letter-spacing: -0.01em;
    }
    .verdict-forged {
        font-size: 3.4rem;
        font-weight: 700;
        color: var(--danger);
        text-align: center;
        font-family: 'DM Serif Display', serif !important;
        text-shadow: 0 0 30px var(--danger-glow);
        letter-spacing: -0.01em;
    }

    /* Header */
    .main-title {
        font-size: 5.5rem;
        font-weight: 400;
        color: var(--gold);
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        font-family: 'DM Serif Display', serif !important;
        text-shadow: 0 0 40px var(--gold-glow);
    }
    .main-subtitle {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-align: center;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Section tags */
    .section-tag {
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 1rem;
        display: block;
        font-weight: 600;
    }
    .upload-label {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
        font-weight: 400;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2.5rem 0;
    }

    /* Info box */
    .info-box {
        background: var(--card);
        border-left: 3px solid var(--gold);
        padding: 1.1rem 1.4rem;
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.85;
        margin-top: 1.1rem;
        margin-bottom: 2rem;
        border-radius: 0 10px 10px 0;
    }
    .info-box strong { color: var(--gold); }

    /* Global overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        max-width: 1400px !important;
    }

    [data-testid="stImage"] > div > p {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
    }

    /* Upload zone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 0 !important;
        padding: 4px 0 !important;
        margin-left: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div {
        background-color: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] > div:first-child {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--gold) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        padding: 0.4rem 1.2rem !important;
        box-shadow: none !important;
        margin-top: 4px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--card) !important;
        border-color: var(--gold) !important;
        transform: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button span { display: none !important; }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Browse file" !important;
        color: var(--gold) !important;
    }
            
    /* File uploader size/type hint text */
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzone"] span {
        color: #8a8a8a !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Remove the uploaded image button */
    [data-testid="stFileChipDeleteBtn"] {
        margin-right: 8px !important;
    }
    [data-testid="stBaseButton-minimal"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px !important;
        color: var(--gold) !important;
    }
    [data-testid="stBaseButton-minimal"] svg {
        width: 18px !important;
        height: 18px !important;
        color: var(--gold) !important;
    }
            
    /* Hide the standalone Browse file button on the right */
    [data-testid="stBaseButton-borderlessIcon"] {
        display: none !important;
    }

    /* Signature preview box */
    .sig-preview-box {
        background: var(--card);
        border: 1px solid var(--gold);
        border-radius: 14px;
        box-shadow: 0 0 18px var(--gold-glow);
        overflow: hidden;
        width: 100%;
        aspect-ratio: 16 / 7;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .sig-preview-box img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 12px;
        padding: 12px;
        box-sizing: border-box;
    }

    /* Empty upload zone */
    .sig-empty-box {
        background: var(--card);
        border: 1px dashed var(--border);
        border-radius: 14px;
        width: 100%;
        aspect-ratio: 16 / 7;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        gap: 0.5rem;
    }
    .sig-empty-box svg { opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# Model download from Google Drive 
MODEL_PATH = "resnet50_signature_final.pth"
DRIVE_FILE_ID = "1ySVNnAGiulOjLCWL9qjayfb2-CT7LoXx"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights... (first run only)"):
        gdown.download(
            f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

OPTIMAL_THRESHOLD = 0.62
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool
        )

    def forward(self, x):
        return self.features(x).flatten(1)


# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        torch.abs(output[0, 0].backward())
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def make_gradcam_overlay(pil_image, cam):
    img_np = np.array(pil_image.resize((224, 224)).convert('RGB'))
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
    return overlay


# Inference
def run_inference(ref_image, query_image, model):
    extractor = FeatureExtractor(model).to(DEVICE)
    extractor.eval()
    gradcam = GradCAM(model, model.layer4[-1])

    ref_tensor = transform(ref_image).unsqueeze(0).to(DEVICE)
    query_tensor = transform(query_image).unsqueeze(0).to(DEVICE)
    query_tensor.requires_grad_(True)

    with torch.no_grad():
        ref_emb = extractor(ref_tensor)
        query_emb = extractor(query_tensor)
    similarity = cosine_similarity(ref_emb, query_emb).item()

    with torch.set_grad_enabled(True):
        output = model(query_tensor)
        classifier_prob = torch.sigmoid(output).item()

    cam = gradcam.generate(query_tensor)
    overlay = make_gradcam_overlay(query_image, cam)

    combined_score = 0.5 * similarity + 0.5 * classifier_prob
    verdict = 'Genuine' if combined_score >= OPTIMAL_THRESHOLD else 'Forged'
    confidence= combined_score if verdict == 'Genuine' else 1 - combined_score

    return {
        'verdict': verdict,
        'confidence': confidence,
        'similarity': similarity,
        'classifier_prob': classifier_prob,
        'combined_score': combined_score,
        'gradcam_overlay': overlay
    }

# Helpers
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sig_preview(img, key_suffix: str):
    if img is not None:
        b64 = pil_to_b64(img)
        st.markdown(f"""
        <div class="sig-preview-box">
            <img src="{b64}" alt="Signature preview">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sig-empty-box">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.5">
              <rect x="3" y="3" width="18" height="18" rx="3"/>
              <path d="M3 9h18M9 21V9"/>
            </svg>
            No signature uploaded
        </div>
        """, unsafe_allow_html=True)


# Page 1: Upload
def page_upload():
    st.markdown("""
    <p class="main-title" style="font-size: 2rem;">SignVerify</p>
    <p class="main-subtitle">Handwritten Signature Forgery Detection — ResNet50</p>
    <hr class="divider">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>SignVerify</strong> uses deep learning to detect forged handwritten signatures. Upload two signatures below to begin analysis.
    </div>
    """, unsafe_allow_html=True)

    col1, col_gap, col2 = st.columns([5, 1, 5])

    ref_file = None
    query_file = None
    ref_image = None
    query_image = None

    with col1:
        st.markdown('<span class="section-tag">01 - Reference Signature</span>', unsafe_allow_html=True)
        st.markdown('<p class="upload-label">Upload a known genuine signature</p>', unsafe_allow_html=True)
        ref_file = st.file_uploader("Reference", type=["png", "jpg", "jpeg"],
                                    label_visibility="collapsed", key="ref")
        if ref_file is not None:
            ref_image = Image.open(ref_file).convert('RGB')
        sig_preview(ref_image, "ref")

    with col2:
        st.markdown('<span class="section-tag">02 - Query Signature</span>', unsafe_allow_html=True)
        st.markdown('<p class="upload-label">Upload the signature to verify</p>', unsafe_allow_html=True)
        query_file = st.file_uploader("Query", type=["png", "jpg", "jpeg"],
                                      label_visibility="collapsed", key="query")
        if query_file is not None:
            query_image = Image.open(query_file).convert('RGB')
        sig_preview(query_image, "query")

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])

    both_files_ready = (ref_file is not None) and (query_file is not None) and model_loaded

    with mid:
        evaluate = st.button(
            "Evaluate Signature",
            disabled=not both_files_ready,
            use_container_width=True
        )

    if evaluate and both_files_ready and ref_image is not None and query_image is not None:
        with st.spinner("Analysing signatures..."):
            st.session_state.ref_image_stored = ref_image
            st.session_state.query_image_stored = query_image
            st.session_state.results = run_inference(ref_image, query_image, model)
        st.session_state.page = 'results'
        st.rerun()

    if not both_files_ready:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0;">
            <p style="font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase; color:#444;">
                Upload both signatures above to begin
            </p>
        </div>
        """, unsafe_allow_html=True)


# Page 2: Results
def page_results():
    st.markdown("""
    <p class="main-title" style="font-size: 2rem;">SignVerify</p>
    <p class="main-subtitle">Handwritten Signature Forgery Detection — ResNet50</p>
    <hr class="divider">
    """, unsafe_allow_html=True)

    results = st.session_state.results
    ref_image = st.session_state.ref_image_stored
    query_image = st.session_state.query_image_stored

    st.markdown('<span class="section-tag">Results</span>', unsafe_allow_html=True)

    res_col1, res_gap, res_col2 = st.columns([4, 1, 7])

    with res_col1:
        verdict_class = "verdict-genuine" if results['verdict'] == 'Genuine' else "verdict-forged"
        verdict_icon = "✓" if results['verdict'] == 'Genuine' else "✗"
        st.markdown(
            f'<p class="{verdict_class}">{verdict_icon} {results["verdict"]}</p>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        conf_pct = results['confidence'] * 100
        sim_pct = results['similarity'] * 100
        cls_pct = results['classifier_prob'] * 100
        comb_pct = results['combined_score'] * 100

        bar_color = "#c9a96e" if results['verdict'] == 'Genuine' else "#cf6679"

        for label, value, pct in [
            ("Confidence", f"{conf_pct:.1f}%", conf_pct),
            ("Similarity Score", f"{sim_pct:.1f}%",  sim_pct),
            ("Classifier Score", f"{cls_pct:.1f}%",  cls_pct),
            ("Combined Score", f"{comb_pct:.1f}%", comb_pct),
        ]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-bar-bg">
                    <div class="metric-bar-fill"
                         style="width:{min(pct,100):.1f}%; background:{bar_color};
                                box-shadow: 0 0 8px {'rgba(201,169,110,0.5)' if bar_color=='#c9a96e' else 'rgba(207,102,121,0.5)'};">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        threshold_note = "above" if results['combined_score'] >= OPTIMAL_THRESHOLD else "below"
        st.markdown(f"""
        <div class="info-box">
            Combined score {results['combined_score']:.4f} is {threshold_note}
            the decision threshold of {OPTIMAL_THRESHOLD:.2f}.<br><br>
            Threshold was tuned to minimise False Acceptance Rate
            while keeping False Rejection Rate under 15%.
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        img_col1, img_col2, img_col3 = st.columns(3)

        for col, img, label in [
            (img_col1, ref_image,                                          'Reference (Genuine)'),
            (img_col2, query_image,                                         'Query'),
            (img_col3, Image.fromarray(results['gradcam_overlay']),         'Model Attention'),
        ]:
            with col:
                st.markdown(
                    f'<p style="font-size:0.68rem; color:#8a8a8a; text-transform:uppercase;'
                    f'letter-spacing:0.12em; margin-bottom:0.4rem;">{label}</p>',
                    unsafe_allow_html=True
                )
                st.image(img, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            The heatmap highlights regions the model focused on when
            making its decision. Red/yellow areas had the highest
            influence on the output. Compare the reference and query
            signatures - structural similarities drive a higher
            similarity score.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr class='divider'>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.page = 'upload'
            st.session_state.results = None
            st.session_state.ref_image_stored = None
            st.session_state.query_image_stored = None
            st.rerun()


# Load model (once, at startup) 
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model: {e}\n\nSet MODEL_PATH to ResNet.")
    model_loaded = False


# MAIN APP LOGIC 
if st.session_state.page == 'upload':
    page_upload()
elif st.session_state.page == 'results':
    page_results()
