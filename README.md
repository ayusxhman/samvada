# 🤟 Samvada — Bidirectional AI Sign Language Translation System

**Samvada** is a real-time, bidirectional AI-powered sign language translation system with multilingual support and hierarchical semantic gloss mapping. It bridges communication between Deaf/hard-of-hearing communities and hearing individuals through two pipelines:

- **Pipeline A (Sign → Text):** Real-time sign language recognition via webcam using MediaPipe landmarks, kinematic feature engineering, and neural sequence models (FingerspellCNN + WordBiGRU).
- **Pipeline B (Text/Speech → Sign):** Converts spoken/typed language to sign language glosses using NLP dependency parsing, 8-tier semantic resolution, and multilingual translation.

## 🏗️ Architecture

| Component | Technology |
|-----------|-----------|
| **Backend** | Python FastAPI (port 8000) |
| **Frontend** | Next.js (React) with static export |
| **Sign Recognition** | MediaPipe Holistic → One-Euro Filter → 7-group Feature Extraction → FingerspellCNN / WordBiGRU |
| **Text-to-Sign** | SpaCy Grammar → 8-Tier Gloss Mapper → Multilingual Translation |
| **Speech** | Groq Whisper Large V3 Turbo / Faster-Whisper fallback |
| **Translation** | 3-tier cascade: Groq Llama 3.1 → Google Translate → NLLB-200 |
| **Languages** | 34 supported languages |

## 📊 Performance

| Model | Accuracy | Details |
|-------|----------|---------|
| **WordBiGRU** | 93.74% Top-1, 98.22% Top-5 | 2,729 word classes, 83,328 sequences |
| **FingerspellCNN** | 98.42% Top-1 | 29 ASL alphabet classes, EfficientNet-B0 backbone |
| **End-to-End Latency** | 35.7 ms (~28 FPS) | Real-time on consumer hardware |

## 📁 Project Structure

```
projectx/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── config.py            # Configuration & paths
│   │   ├── models/              # Neural network architectures
│   │   │   ├── fingerspell_model.py  # EfficientNet-B0 CNN
│   │   │   └── word_model.py         # 4-layer BiGRU + self-attention
│   │   ├── services/            # Core business logic
│   │   │   ├── sign_to_text.py       # Sign recognition pipeline
│   │   │   ├── text_to_sign.py       # Text-to-sign translation
│   │   │   ├── translation.py        # 3-tier translation cascade
│   │   │   ├── speech_service.py     # Speech recognition
│   │   │   ├── gloss_mapper.py       # 8-tier semantic resolver
│   │   │   ├── grammar.py            # English→ASL grammar
│   │   │   ├── feature_extractor.py  # 7-group kinematic features
│   │   │   ├── landmark_stabilizer.py # One-Euro filter
│   │   │   └── dictionary.py         # Dynamic word learning
│   │   ├── routers/             # API route handlers
│   │   └── orchestration/       # Central AI coordinator
│   ├── trained_models/          # Pre-trained model weights
│   │   ├── word_model.pth       # WordBiGRU weights
│   │   ├── fingerspell_model.pth # FingerspellCNN weights
│   │   └── word_labels.json     # Class label mappings
│   └── training/                # Training scripts
│       └── train_word_recognition.py
├── frontend/
│   ├── src/app/
│   │   └── page.tsx             # Main UI page
│   ├── public/                  # Static assets
│   └── package.json
├── dataset/                     # Dataset configs (data downloaded separately)
│   ├── dataset_manifest.json
│   ├── complex_synonyms_map.json
│   ├── normalization_stats.json
│   └── per_word_accuracy.csv
├── paper_figures/               # IEEE paper figures
├── generate_ieee_paper.py       # IEEE paper generation script
├── generate_figures.py          # Figure generation script
├── run.py                       # Application launcher
└── start.bat                    # Windows quick-start
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python start.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run build
```

### Run Everything

```bash
# Windows
start.bat

# Or manually
python run.py
```

The backend runs on `http://localhost:8000` and serves the frontend static files.

## 📦 Datasets

The datasets are too large (~125 GB) for GitHub. Download them separately:

| Dataset | Source | Samples |
|---------|--------|---------|
| ASL Alphabet | [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 138,118 |
| ASL Citizen | [ASL Citizen Project](https://www.microsoft.com/en-us/research/project/asl-citizen/) | 83,399 |
| Hindi SL | ISL Research Corpus | 48,000 |
| How2Sign | [How2Sign](https://how2sign.github.io/) | 35,263 |
| WLASL | [WLASL GitHub](https://github.com/dxli94/WLASL) | 21,083 |
| ISL-CSLTR | ISL Continuous SLR | 21,074 |
| **Total** | | **346,937** |

Place downloaded datasets in the `dataset/` directory following the structure in `dataset_manifest.json`.

## 📄 Research Paper

The IEEE research paper is generated via:

```bash
python generate_figures.py   # Generate paper figures
python generate_ieee_paper.py # Generate the DOCX
```

Output: `Samvada_IEEE_Research_Paper_Final.docx`

## 🔑 API Keys

Configure your Groq API key in `backend/app/config.py` or set the `GROQ_API_KEY` environment variable.

## 📜 License

This project was developed for academic research purposes.

## 👥 Authors

Ayushman and Team
