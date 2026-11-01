# NewsLens AI — News Article Classification Using Deep Learning

A full-stack educational project for classifying news into Politics, Sports, Business, Technology, Entertainment, Health, and Science.

## Included features
- User registration, login/logout, salted PBKDF2 password hashing and session-protected pages
- Single article classification from pasted text or public article URL (URL extraction is best-effort)
- Prediction probabilities, source URL storage, searchable history, saved articles and category feedback
- CSV batch classification and downloadable results
- Analytics dashboard with category chart and summary metrics
- Model Lab with **MLP baseline** and optional **CNN, LSTM, BiLSTM and BERT** model choices
- Model evaluation: accuracy, precision, recall, macro-F1, classification report and confusion matrix
- Dataset upload in CSV format for model training
- SQLite persistence and JSON classification API
- Clear fallback when optional deep-learning packages are not installed

## Run on Windows / VS Code
1. Install Python 3.10 or 3.11 (recommended for optional TensorFlow compatibility).
2. Extract ZIP and open this folder in VS Code.
3. Terminal:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```
4. Open http://127.0.0.1:5000 and register an account.
5. Open Model Lab and train the MLP baseline. The first run trains automatically if no model exists.

## Optional models
The MLP model works with the base requirements. For CNN/LSTM/BiLSTM:
```powershell
pip install tensorflow
```
For BERT (transformer):
```powershell
pip install torch transformers
```
Then restart the app and use Model Lab. BERT may download pretrained weights on first use and requires more RAM/disk/network access. If a package/model is unavailable, the UI reports the dependency or download issue rather than pretending the model ran.

## Dataset format
Upload CSV with a text/content/article/news column and a category/label column. Optional title/headline columns are supported.
Example:
```csv
title,text,category
"Team wins final","The team won the championship after a close match.",Sports
"Company reports growth","The company reported increased quarterly revenue.",Business
```
Use a licensed, representative dataset for research. The bundled `data/news_dataset_2000.csv` contains 2,000 **synthetic demonstration records** balanced across seven categories (286 records in five categories and 285 in two). It is used by default for model training. These are generated examples, not real news reports; use a reputable real-world dataset for research claims.

## URL extraction note
Public URLs are fetched only when accessible. Publisher sites may block bots, require JavaScript/login, or restrict access. In those cases paste article text manually. Use only content you are permitted to access.

## API
`POST /api/classify` (requires a logged-in session cookie):
```json
{"title":"New AI tool launched","text":"Researchers introduced a machine learning platform."}
```

## Models and algorithms
- TF-IDF + MLPClassifier baseline (runs with base requirements)
- Optional Keras text CNN
- Optional Keras LSTM
- Optional Keras BiLSTM
- Optional Hugging Face BERT sequence classifier
- Train/test split, stratification when possible, accuracy, precision/recall/F1, confusion matrix
- Password PBKDF2 hashing, SQLite database, session auth, CSV parsing and best-effort HTML text extraction

## Production cautions
This is an academic prototype. Before public deployment add CSRF protection, rate limiting, secure cookie settings, a production WSGI server, stronger SSRF protection, and model/dataset governance. Never treat predictions as verified facts.


## Fixes in this updated package
- Fixed the home page route so it renders the landing page instead of redirecting to HTML as a URL.
- The Model Lab now uses `data/news_dataset_2000.csv` by default when training MLP, CNN, LSTM, BiLSTM, or BERT (where optional dependencies are installed), rather than silently falling back to tiny demo samples.
- The bundled 2,000-row dataset is synthetic for coursework/demo use. It is not a collection of verified real-world news articles; evaluate on a trustworthy real dataset before making research or production claims.

## Run on Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000/` in your browser, register an account, sign in, and open **Model Lab** to train the baseline model.

## Dataset browser
After signing in, choose **Dataset (2,000)** in the sidebar or visit `/dataset`. The page displays the total corpus count, category distribution, searchable records, and 25 records per page. This dataset is distinct from the dashboard's personal "Total analyzed" count, which only increases when the signed-in user classifies articles.

The bundled dataset has 2,000 synthetic demonstration records across seven categories. It is not verified real-world news.

## Model algorithm update
This release removes misleading optional TensorFlow/BERT choices from the normal interface. The following classifiers work with the standard requirements: Hybrid, MLP, Logistic Regression, calibrated Linear SVM, and Multinomial Naive Bayes. Each selected model is trained and saved independently; predictions automatically train the selected model if it has not yet been trained. The Hybrid model combines word/character TF-IDF Logistic Regression probabilities with a small transparent domain-keyword prior.

The default prediction is Hybrid. The bulk classifier also uses Hybrid. URL extraction is best-effort; if a publisher blocks extraction, paste the article headline and full article body manually. A top probability is a model score, not guaranteed accuracy.

The included 2,000-record dataset is synthetic demonstration data. It should not be described as a verified news corpus or used alone to claim real-world accuracy.

## Validation performed for this release
- Confirmed the CSV contains exactly 2,000 rows.
- Confirmed Hybrid, Logistic Regression, calibrated Linear SVM, Multinomial Naive Bayes, and MLP train successfully using the standard requirements.
- Smoke-tested all five classifiers with the supplied Rahul Gandhi / vote-chori political article text; all five returned Politics in this test.
- The bundled synthetic corpus is highly patterned. A random holdout split can produce inflated scores (including 1.00 on this generated corpus); this is not evidence of 100% real-world accuracy. For a defensible project evaluation, replace or supplement it with real, independently collected, labeled news articles and use a publisher/source-separated or time-separated test set.
