# 🎨 Anime GAN Showdown - Streamlit App

A creative and interactive Streamlit application that demonstrates and compares **DCGAN vs WGAN-GP** for anime face generation.

## ⚡ Quick Deploy

**Deploy on Streamlit Cloud (1 click):**
1. Go to https://streamlit.io/cloud
2. Connect your GitHub repo
3. Select this repo branch and `anime_gan_app.py` as main file
4. Click Deploy! 🚀

Public link will be live in 2-3 minutes!

## 🎯 Features

### 🎲 **Generation Tab**
- Generate anime faces with DCGAN
- Generate anime faces with WGAN-GP
- Control number of samples (1-16)
- Reproducible with seed control

### 📊 **Comparison Tab**
- Side-by-side model comparison
- Real-time generation

### 🔍 **Deep Dive Tab**
- Architecture details
- Loss function explanations
- Model parameters

### 📈 **Training Insights Tab**
- Interactive loss curves
- Training metrics

### ⚡ **Key Differences Tab**
- Model comparison table
- Pros/cons analysis

## 🏃 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run anime_gan_app.py
```

App opens at `http://localhost:8501`

## 🤖 Models

### DCGAN
- **Loss:** Binary Cross-Entropy
- **Epochs:** 50
- **Training Time:** ~24 hours
- **Status:** Simple, but potentially unstable

### WGAN-GP  
- **Loss:** Wasserstein Distance + Gradient Penalty
- **Epochs:** 30
- **Training Time:** ~18 hours
- **Status:** More stable, better convergence

## 📊 Architecture

Both models:
- **Input:** 100D random noise vector
- **Output:** 64×64 RGB anime faces
- **Feature Maps:** 64

## 🔑 Key Differences

| Aspect | DCGAN | WGAN-GP |
|--------|-------|---------|
| Loss Function | BCE | Wasserstein + GP |
| Convergence | Slow & Unstable | Fast & Stable |
| Mode Collapse | Prone | Mitigated |
| Training Stability | Requires Tuning | Robust |

## 📁 Project Structure

```
DCGAN-vs-WGAN-GP/
├── anime_gan_app.py          # Main Streamlit app
├── utils.py                  # Helper functions
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── genai-ass3-q1.ipynb      # Training notebook
```

## 📦 Requirements

- Python 3.8+
- streamlit
- torch & torchvision
- numpy, matplotlib, plotly
- Pillow

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
```
1. Push to GitHub ✓ (you're here!)
2. Go to streamlit.io/cloud
3. Connect repo
4. Deploy
```

### Local Machine
```bash
pip install -r requirements.txt
streamlit run anime_gan_app.py
```

## 📚 References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [Streamlit Docs](https://docs.streamlit.io)
- [PyTorch Docs](https://pytorch.org)

## 💡 Usage

1. Navigate to the live app
2. Use sidebar to configure parameters
3. Click tabs to explore different features
4. Generate images and compare models
5. View training insights and analysis

---

**Status:** ✅ Ready for Deployment  
**Version:** 1.0  
**Created:** 2024
