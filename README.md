# 🌸 Rangoli Pattern AI

> An AI-powered web application that generates and analyzes traditional Indian **Rangoli patterns** using geometric symmetry algorithms, graph theory, and computer vision.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎨 **5 Pattern Styles** | Mandala, Kolam, Floral, Geometric, Peacock |
| 🎛️ **Customizable** | Symmetry folds, layers, complexity, color scheme |
| 🔬 **Image Analyzer** | Upload any Rangoli image to detect symmetry, colors & shapes |
| 🕸️ **Graph Representation** | Delaunay triangulation + NetworkX graph statistics |
| ⬇️ **Download** | Export generated patterns as high-res PNG |
| 📱 **Mobile-Responsive** | Works on all screen sizes |

---

## 🚀 Quick Start (Local)

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/rangoli-pattern-ai.git
cd rangoli-pattern-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Edit `.env` and set a strong `SECRET_KEY`.

### 3. Run the development server

```bash
python app.py
```

Visit **http://localhost:5000** in your browser.

---

## 🏭 Production Deployment

### Option A — Render.com (Recommended, Free)

1. Push the project to a GitHub repository.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your GitHub repository.
4. Render auto-detects `render.yaml` and configures the service.
5. Add `SECRET_KEY` in the Render environment variables dashboard.
6. Click **Deploy**.

### Option B — Railway

1. Install the [Railway CLI](https://docs.railway.app/develop/cli): `npm i -g @railway/cli`
2. `railway login && railway init && railway up`
3. Set environment variables via the Railway dashboard.

### Option C — Heroku

```bash
heroku create your-rangoli-app
heroku config:set SECRET_KEY=your-super-secret-key FLASK_ENV=production
git push heroku main
```

### Option D — Manual (Gunicorn)

```bash
gunicorn wsgi:application --workers=2 --timeout=120 --bind=0.0.0.0:8000
```

---

## 🔌 API Reference

### `POST /api/generate`
Generate a Rangoli pattern image.

**Request body (JSON):**
```json
{
  "style": "mandala",
  "symmetry": 8,
  "layers": 5,
  "colorScheme": "traditional",
  "complexity": 50
}
```

**Response:**
```json
{
  "success": true,
  "image": "<base64-encoded PNG>",
  "graph_analysis": {
    "nodes": 120,
    "edges": 340,
    "density": 0.047,
    "avg_degree": 5.67,
    "connected_components": 1,
    "symmetry_order": 8
  }
}
```

---

### `POST /api/analyze`
Analyze an uploaded Rangoli image.

**Request:** `multipart/form-data` with field `image` (PNG/JPG).

**Response:**
```json
{
  "success": true,
  "symmetry_analysis": { "2-fold": 91.2, "dominant_symmetry": "8-fold", ... },
  "color_palette": [{ "rgb": "rgb(255,107,53)", "hex": "#ff6b35", "percentage": 32.1 }],
  "pattern_features": { "lines_detected": 48, "complexity_score": 72, ... },
  "edge_map": "<base64-encoded PNG>"
}
```

---

### `GET /api/styles`
Returns all available styles, color schemes, and configuration options.

---

### `GET /health`
Health-check endpoint — returns `{"status": "ok"}`.

---

## 🧠 Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, Flask 3.0, Gunicorn |
| **Image Generation** | Pillow (PIL), NumPy, matrix geometry |
| **Computer Vision** | OpenCV (Canny edges, Hough transforms) |
| **Graph Theory** | NetworkX, SciPy Delaunay triangulation |
| **Machine Learning** | scikit-learn K-Means clustering |
| **Frontend** | Vanilla HTML/CSS/JS, Google Fonts |

---

## 📁 Project Structure

```
RANGOLI/
├── app.py              # Flask backend + all ML/CV logic
├── wsgi.py             # WSGI entry point for production
├── requirements.txt    # Python dependencies
├── Procfile            # Heroku/Railway process definition
├── render.yaml         # Render.com deployment config
├── .env.example        # Environment variable template
├── .gitignore
├── templates/
│   └── index.html      # Single-page application
└── static/
    └── style.css       # Premium dark-mode design system
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
