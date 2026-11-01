
import os, json, pickle, re
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT=os.path.dirname(os.path.abspath(__file__))
MODELS=os.path.join(ROOT,"models")
DATA=os.path.join(ROOT,"data")
CATEGORIES=["Politics","Sports","Business","Technology","Entertainment","Health","Science"]
MODEL_FILES={
    "MLP":"mlp_pipeline.pkl",
    "LOGISTIC":"logistic_pipeline.pkl",
    "LINEAR_SVM":"linear_svm_pipeline.pkl",
    "NAIVE_BAYES":"naive_bayes_pipeline.pkl",
    "HYBRID":"hybrid_pipeline.pkl",
}
def load_dataset(path=None):
    if not path:
        path=os.path.join(DATA,"news_dataset_2000.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("Training dataset not found: "+str(path))
    import pandas as pd
    df=pd.read_csv(path).fillna("")
    cols={str(c).lower().strip():c for c in df.columns}
    textc=next((cols[x] for x in ("text","content","article","news","description") if x in cols),None)
    titlec=next((cols[x] for x in ("title","headline") if x in cols),None)
    labelc=next((cols[x] for x in ("category","label","class","topic") if x in cols),None)
    if not textc or not labelc:
        raise ValueError("CSV must contain text/content/article and category/label columns.")
    texts=[]; labels=[]
    for _,r in df.iterrows():
        parts=[]
        if titlec and str(r[titlec]).strip(): parts.append(str(r[titlec]).strip())
        if str(r[textc]).strip(): parts.append(str(r[textc]).strip())
        tx=" ".join(parts); lab=str(r[labelc]).strip()
        if len(tx)>=15 and lab: texts.append(tx); labels.append(lab)
    if len(texts)<20 or len(set(labels))<2:
        raise ValueError("Dataset needs at least 20 valid records and two categories.")
    return texts,labels

def build_features():
    return FeatureUnion([
        ("word",TfidfVectorizer(ngram_range=(1,2),max_features=30000,stop_words="english",sublinear_tf=True,strip_accents="unicode")),
        ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=25000,sublinear_tf=True,min_df=2))
    ])

def make_pipeline(name):
    features=build_features()
    if name in ("HYBRID","LOGISTIC"):
        clf=LogisticRegression(C=4.0,max_iter=1200,class_weight="balanced",solver="lbfgs")
    elif name=="LINEAR_SVM":
        base=LinearSVC(C=1.5,class_weight="balanced")
        clf=CalibratedClassifierCV(base,cv=3)
    elif name=="NAIVE_BAYES":
        return Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=30000,stop_words="english",sublinear_tf=True)),("clf",MultinomialNB(alpha=.12))])
    elif name=="MLP":
        clf=MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=.001,max_iter=140,early_stopping=False,random_state=42)
    else: raise ValueError("Unknown model: "+name)
    return Pipeline([("features",features),("clf",clf)])

def train_model(model_name="HYBRID",csv_path=None):
    name=model_name.upper()
    if name not in MODEL_FILES: raise ValueError("Choose HYBRID, MLP, LOGISTIC, LINEAR_SVM or NAIVE_BAYES.")
    texts,labels=load_dataset(csv_path)
    counts={x:labels.count(x) for x in set(labels)}
    strat=labels if min(counts.values())>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(texts,labels,test_size=.20,random_state=42,stratify=strat)
    pipe=make_pipeline(name)
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    labs=sorted(set(labels))
    p,r,f,_=precision_recall_fscore_support(yte,pred,average="macro",zero_division=0)
    metrics={
      "model":name,"accuracy":float(accuracy_score(yte,pred)),
      "precision_macro":float(p),"recall_macro":float(r),"f1_macro":float(f),
      "test_samples":len(yte),"training_samples":len(ytr),"dataset_samples":len(texts),
      "classes":labs,"report":classification_report(yte,pred,labels=labs,output_dict=True,zero_division=0),
      "confusion_matrix":confusion_matrix(yte,pred,labels=labs).tolist(),
      "note":"Holdout split from bundled/supplied dataset. Bundled records are synthetic demonstration data; scores do not establish real-world accuracy."
    }
    os.makedirs(MODELS,exist_ok=True)
    with open(os.path.join(MODELS,MODEL_FILES[name]),"wb") as f: pickle.dump(pipe,f)
    with open(os.path.join(MODELS,name.lower()+"_metrics.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    if name=="HYBRID":
        with open(os.path.join(MODELS,"metrics.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    return metrics

def train_mlp(csv_path=None): return train_model("MLP",csv_path)
def train_optional(model_name,csv_path=None): return train_model(model_name,csv_path)
def evaluate():
    path=os.path.join(MODELS,"metrics.json")
    if not os.path.exists(path): return train_model("HYBRID")
    with open(path,encoding="utf-8") as f:return json.load(f)

# Domain terms supplement statistical classification. They are not fact-checking.
DOMAIN_TERMS={
"Politics":["election","vote","voter","parliament","lok sabha","rajya sabha","rahul gandhi","narendra modi","pm modi","amit shah","bjp","congress","minister","government","opposition","chief election commissioner","cec","polling","ballot","campaign","political","politics","legislature","mla","mp","cabinet","policy","vote chori","electoral","constituency","rally","protest","democracy","president","prime minister"],
"Sports":["cricket","football","soccer","tennis","basketball","olympic","olympics","tournament","championship","match","player","coach","wicket","goal","innings","batsman","bowler","athlete","league","medal","stadium","fifa","ipl","world cup"],
"Business":["stock","shares","market","investor","investment","company","corporate","revenue","profit","loss","earnings","startup","funding","economy","economic","bank","rbi","inflation","interest rate","ipo","trade","business","ceo","acquisition","merger","quarterly","finance"],
"Technology":["software","hardware","artificial intelligence","ai model","machine learning","smartphone","app","application","cybersecurity","cyber attack","data breach","computer","chip","processor","semiconductor","cloud","robot","robotics","algorithm","technology","tech firm","internet","google","microsoft","apple","openai"],
"Entertainment":["film","movie","actor","actress","cinema","box office","director","singer","concert","album","music","television","tv show","streaming","series","celebrity","festival","award","ott","netflix"],
"Health":["hospital","doctor","patient","disease","medicine","medical","health","healthcare","vaccine","vaccination","clinical trial","symptom","treatment","diagnosis","virus","infection","nutrition","mental health","public health"],
"Science":["scientist","researcher","research study","experiment","laboratory","lab","astronomer","astronomy","physics","chemistry","biology","climate","planet","space telescope","fossil","species","genome","scientific","discovery","scientific study"]
}
def keyword_prior(text):
    t=re.sub(r"\s+"," ",text.lower())
    scores={}
    for cat,terms in DOMAIN_TERMS.items():
        score=0.0
        for term in terms:
            if term in t:
                score += 2.0 if " " in term else 1.0
        scores[cat]=score
    return scores

def predict(text,model_name="HYBRID"):
    name=model_name.upper()
    if name not in MODEL_FILES: raise RuntimeError("Choose HYBRID, MLP, LOGISTIC, LINEAR_SVM or NAIVE_BAYES.")
    path=os.path.join(MODELS,MODEL_FILES[name])
    if not os.path.exists(path):
        train_model(name)
    with open(path,"rb") as f: pipe=pickle.load(f)
    probs=pipe.predict_proba([text])[0]
    classes=list(pipe.classes_)
    scores={str(c):float(p) for c,p in zip(classes,probs)}
    if name=="HYBRID":
        prior=keyword_prior(text)
        # A strong multi-term domain match supplements the learned model.
        if prior:
            best=max(prior,key=prior.get); strength=prior[best]
            if strength>=2:
                denom=sum(prior.values()) or 1
                rule={c:(prior.get(c,0)/denom) for c in classes}
                blend=min(.55,.18+.035*strength)
                scores={c:(1-blend)*scores[c]+blend*rule.get(c,0) for c in classes}
    total=sum(scores.values()) or 1
    ranked=sorted(((k,v/total) for k,v in scores.items()),key=lambda x:x[1],reverse=True)
    return {"category":ranked[0][0],"confidence":float(ranked[0][1]),"model":name,
            "probabilities":[{"category":k,"probability":float(v)} for k,v in ranked],
            "confidence_note":"Model score, not a guarantee of correctness. Review low-confidence predictions."}
