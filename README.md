# ✦ NewsLens AI
### Decode the News. Discover the Topic. Understand the Story.

<p align="center">
  <b>An AI-powered News Article Classification Platform using Machine Learning and Deep Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Database-SQLite-blue?logo=sqlite" />
  <img src="https://img.shields.io/badge/Status-Academic%20Project-success" />
</p>

<p align="center">
  <a href="YOUR_DEPLOYED_WEBSITE_URL">🚀 Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-project-architecture">Architecture</a>
</p>

---

## 🌐 Project Overview

**NewsLens AI** is an intelligent news classification platform designed to automatically identify the primary topic of a news article using Natural Language Processing (NLP), Machine Learning and Deep Learning techniques.

In a world where thousands of news articles are published every day, manually organizing news content is time-consuming.

NewsLens AI simplifies this process by analysing article headlines and content, extracting meaningful text features, and predicting the most relevant news category.

### 🎯 Seven News Categories

| Category | Description |
|---|---|
| 🏛️ Politics | Elections, government, policies and political developments |
| 🏏 Sports | Cricket, football, tournaments and sports events |
| 💼 Business | Finance, markets, companies and economy |
| 💻 Technology | AI, software, cybersecurity and innovation |
| 🎬 Entertainment | Movies, music, celebrities and television |
| 🏥 Health | Healthcare, medicine and public health |
| 🔬 Science | Research, discoveries, space and scientific developments |

---

## ✨ Features

### 🧠 Intelligent Article Classification
- Classify news articles using their headline and content.
- Extract article text from publicly accessible URLs.
- Predict the most likely news category.
- Display category-wise probability scores.
- Compare predictions across available models.

### 📊 Interactive Dashboard
- Total articles analysed.
- Saved articles.
- Average prediction confidence.
- Category distribution.
- News classification activity.

### 📁 Dataset Management
- Upload custom CSV datasets.
- Train models using labelled news articles.
- View training and evaluation results.
- Support for headline, article text and category columns.

### ⚡ Bulk News Classification
- Upload multiple articles using CSV.
- Automatically classify each article.
- View category predictions and confidence scores.
- Export classification results.

### 👤 User Management
- User registration and login.
- Session-based authentication.
- Password hashing.
- Personal classification history.

### 📚 History & Saved Articles
- Browse previously analysed articles.
- Save articles for later reference.
- Search classification history.
- Submit feedback on predicted categories.

### 🧪 Model Lab
Train and evaluate the available classification models.

- MLP + TF-IDF
- CNN
- LSTM
- BiLSTM
- BERT Transformer (optional dependencies required)

Evaluation includes:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Classification report
- Confusion matrix

---

## 🏗️ Project Architecture

```text
                 NEWS ARTICLE INPUT
                         |
              +----------+----------+
              |                     |
          Article URL          Manual Text
              |                     |
              +----------+----------+
                         |
                  TEXT EXTRACTION
                         |
                  TEXT PREPROCESSING
                         |
                 FEATURE EXTRACTION
                      TF-IDF
                         |
                MACHINE LEARNING
                         |
             +-----------+-----------+
             |           |           |
            MLP         CNN        LSTM
             |           |           |
             +-----------+-----------+
                         |
                 CLASSIFICATION
                         |
                CATEGORY PREDICTION
                         |
              +----------+----------+
              |                     |
        Probability Scores     Result Storage
              |                     |
              +----------+----------+
                         |
                  WEB DASHBOARD
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap / custom styling |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras (optional models) |
| NLP | TF-IDF, text preprocessing |
| Transformer | PyTorch, Hugging Face Transformers (optional BERT) |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualisation | Chart.js / dashboard charts |

---

## 📂 Project Structure

```text
NewsLens-AI/
│
├── app.py
├── model_service.py
├── requirements.txt
├── README.md
│
├── data/
│   └── news_dataset_2000.csv
│
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── classify.html
│   ├── bulk.html
│   ├── history.html
│   └── model_lab.html
│
├── static/
│   └── style.css
│
└── .gitignore
```

---

## 📊 Dataset Information

The project includes a labelled news dataset for demonstrating multi-class text classification.

**Dataset fields:**

| Column | Description |
|---|---|
| id | Unique record identifier |
| title | News headline |
| text | News article content |
| category | Target classification label |
| source_url | Article source URL, if available |

**Supported categories:** Politics, Sports, Business, Technology, Entertainment, Health and Science.

The bundled 2,000-record dataset contains synthetic demonstration records. It is intended for application development and testing, not as a verified real-world news corpus.

Custom labelled CSV files can also be uploaded through Model Lab.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajesh-r-16/NewsLens-AI.git
```

### 2. Open the project

```bash
cd NewsLens-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in browser

```text
http://127.0.0.1:5000
```

Register an account, log in and start analysing news articles.

---

## 🚀 Deployment

The Flask application can be deployed using Render.

**Deployment platform:** [Render](https://render.com/)

### Deployment configuration

| Setting | Value |
|---|---|
| Runtime | Python |
| Branch | main |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app` |

Add a secure `SECRET_KEY` environment variable in your hosting dashboard.

**Live application:** [Open NewsLens AI](https://newslens-ai-ydpb.onrender.com)



For production, configure persistent database storage, secure cookies, CSRF protection and appropriate model storage.

---

## 🔮 Future Enhancements

- Integration of larger real-world news datasets.
- Improved model performance through hyperparameter tuning.
- Multilingual news classification.
- Fake news and misinformation detection.
- Sentiment analysis.
- Real-time news collection and classification.
- Explainable AI for prediction reasoning.
- Advanced model comparison dashboard.
- Cloud database integration.

---

## ⚠️ Limitations

- URL extraction depends on publisher accessibility; some sites block automated requests.
- Predictions are model estimates and are not verified facts.
- Synthetic training data cannot establish real-world classification accuracy.
- Optional deep-learning models require additional dependencies and training resources.

---

## 🎓 Project Information

**Project Title:** News Article Classification Using Machine Learning and Deep Learning

**Project Domain:** Artificial Intelligence | Machine Learning | Natural Language Processing

**Project Type:** Full-Stack Web Application

**Repository:** [NewsLens AI – GitHub](https://github.com/rajesh-r-16/NewsLens-AI)

---

## 👨‍💻 Author

**Rajesh R**

Computer Science and Engineering – Artificial Intelligence & Machine Learning

GitHub: [@rajesh-r-16](https://github.com/rajesh-r-16)

---

<p align="center">
  <b>✦ NewsLens AI — Your News, Decoded.</b>
  <br/>
  Turning unstructured news into structured intelligence.
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
