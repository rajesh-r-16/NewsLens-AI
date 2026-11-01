import os, io, csv, json, sqlite3, hashlib, secrets, re, urllib.request, urllib.parse, html
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from model_service import predict, train_mlp, train_optional, evaluate, CATEGORIES
ROOT=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(ROOT,"news_classifier.db")
app=Flask(__name__); app.secret_key=os.environ.get("SECRET_KEY","replace-before-deployment")
app.config["MAX_CONTENT_LENGTH"]=10*1024*1024
def conn():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT UNIQUE,email TEXT UNIQUE,password_hash TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY,user_id INTEGER,title TEXT,content TEXT,source_url TEXT,category TEXT,confidence REAL,probabilities TEXT,model TEXT,created_at TEXT,saved INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY,article_id INTEGER,user_id INTEGER,correct_category TEXT,created_at TEXT);
        """)
init_db()
def pw_hash(pw,salt=None):
    salt=salt or secrets.token_hex(16)
    return salt+"$"+hashlib.pbkdf2_hmac("sha256",pw.encode(),salt.encode(),240000).hex()
def pw_check(pw,stored):
    try:
        salt,digest=stored.split("$",1); return secrets.compare_digest(pw_hash(pw,salt).split("$",1)[1],digest)
    except Exception:return False
def login_required(fn):
    @wraps(fn)
    def w(*a,**kw):
        if "uid" not in session: return redirect(url_for("login"))
        return fn(*a,**kw)
    return w
@app.context_processor
def globals_():return {"categories":CATEGORIES,"current_user":session.get("username")}
@app.route("/")
def home():
    if session.get("uid"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")
@app.route("/register",methods=["GET","POST"])
def register():
    if request.method=="POST":
        u=request.form.get("username","").strip(); e=request.form.get("email","").strip().lower(); p=request.form.get("password","")
        if len(u)<3 or "@" not in e or len(p)<8:flash("Enter a username (3+ chars), valid email, and password (8+ chars).","danger")
        else:
            try:
                with conn() as c:c.execute("INSERT INTO users(username,email,password_hash,created_at) VALUES(?,?,?,?)",(u,e,pw_hash(p),datetime.now().isoformat(timespec="seconds")))
                flash("Account created. Please log in.","success");return redirect(url_for("login"))
            except sqlite3.IntegrityError:flash("Username or email already exists.","danger")
    return render_template("register.html")
@app.route("/login",methods=["GET","POST"])
def login():
    if request.method=="POST":
        x=request.form.get("username","").strip()
        with conn() as c:r=c.execute("SELECT * FROM users WHERE username=? OR email=?",(x,x.lower())).fetchone()
        if r and pw_check(request.form.get("password",""),r["password_hash"]):
            session.clear();session["uid"]=r["id"];session["username"]=r["username"];return redirect(url_for("dashboard"))
        flash("Invalid login details.","danger")
    return render_template("login.html")
@app.route("/logout")
def logout():session.clear();return redirect(url_for("home"))
def fetch_url(url):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme not in ("http","https") or not parsed.netloc:raise ValueError("Enter a valid http(s) article URL.")
    host=(parsed.hostname or "").lower()
    if host in ("localhost","127.0.0.1","::1") or host.endswith(".local"):raise ValueError("Local URLs are not supported.")
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 NewsLens educational classifier"})
        with urllib.request.urlopen(req,timeout=10) as r:
            if "html" not in r.headers.get("Content-Type","") and "text/plain" not in r.headers.get("Content-Type",""):raise ValueError("URL is not an HTML/text page.")
            raw=r.read(1500000).decode("utf-8",errors="replace")
    except Exception as e:raise ValueError("Could not fetch this page. Publisher may block extraction; paste text manually.") from e
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(raw,"html.parser")
        for n in soup(["script","style","nav","footer","header","aside","noscript"]):n.decompose()
        title=soup.title.get_text(" ",strip=True) if soup.title else ""
        blocks=soup.find_all(["article","main"]); target=max(blocks,key=lambda n:len(n.get_text(" ",strip=True))) if blocks else (soup.body or soup)
        text=re.sub(r"\s+"," ",target.get_text(" ",strip=True)).strip()
    except ImportError:
        m=re.search(r"<title[^>]*>(.*?)</title>",raw,re.I|re.S);title=html.unescape(re.sub("<[^>]+>"," ",m.group(1))).strip() if m else ""
        text=html.unescape(re.sub(r"(?s)<[^>]+>"," ",raw));text=re.sub(r"\s+"," ",text).strip()
    if len(text)<80:raise ValueError("Not enough article text found. Paste the article content manually.")
    return title[:300],text[:50000]
def store(title,text,url,result):
    with conn() as c:
        cur=c.execute("INSERT INTO articles(user_id,title,content,source_url,category,confidence,probabilities,model,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
          (session["uid"],title,text,url,result["category"],result["confidence"],json.dumps(result["probabilities"]),result["model"],datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid
@app.route("/dataset")
@login_required
def dataset_records():
    csv_path=os.path.join(ROOT,"data","news_dataset_2000.csv")
    records=[]
    with open(csv_path,encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        for row in reader:
            records.append(row)
    total=len(records)
    counts={category:sum(1 for row in records if row.get("category")==category) for category in CATEGORIES}
    query=request.args.get("q","").strip().lower()
    if query:
        records=[row for row in records if query in " ".join(str(v) for v in row.values()).lower()]
    page=max(1,request.args.get("page",1,type=int))
    per_page=25
    pages=max(1,(len(records)+per_page-1)//per_page)
    page=min(page,pages)
    shown=records[(page-1)*per_page:page*per_page]
    return render_template("dataset.html",records=shown,total=total,filtered=len(records),
                           counts=counts,page=page,pages=pages,q=query)

@app.route("/dashboard")
@login_required
def dashboard():
    with conn() as c:
        rows=c.execute("SELECT * FROM articles WHERE user_id=? ORDER BY id DESC LIMIT 8",(session["uid"],)).fetchall()
        total=c.execute("SELECT COUNT(*) n FROM articles WHERE user_id=?",(session["uid"],)).fetchone()["n"]
        saved=c.execute("SELECT COUNT(*) n FROM articles WHERE user_id=? AND saved=1",(session["uid"],)).fetchone()["n"]
        avg=c.execute("SELECT AVG(confidence) n FROM articles WHERE user_id=?",(session["uid"],)).fetchone()["n"] or 0
        counts={r["category"]:r["n"] for r in c.execute("SELECT category,COUNT(*) n FROM articles WHERE user_id=? GROUP BY category",(session["uid"],))}
    return render_template("dashboard.html",recent=rows,total=total,saved=saved,avg=avg*100,counts=counts)
@app.route("/classify",methods=["GET","POST"])
@login_required
def classify():
    result=None
    if request.method=="POST":
        title=request.form.get("title","").strip();text=request.form.get("content","").strip();url=request.form.get("source_url","").strip();model=request.form.get("model","HYBRID").upper()
        if url:
            try:
                fetched_title,fetched_text=fetch_url(url);title=title or fetched_title;text=text or fetched_text
            except ValueError as e:flash(str(e),"warning")
        if len((title+" "+text).strip())<15:flash("Provide a public article URL that can be extracted, or paste article text. Content is optional only when URL extraction succeeds.","warning")
        else:
            try:
                result=predict((title+" "+text).strip(),model);result["article_id"]=store(title,text or title,url,result);result["source_url"]=url
            except Exception as e:flash(str(e),"danger")
    return render_template("classify.html",result=result)
@app.route("/bulk",methods=["GET","POST"])
@login_required
def bulk():
    rows=None
    if request.method=="POST":
        f=request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):flash("Upload a CSV file.","danger")
        else:
            try:
                rd=csv.DictReader(io.StringIO(f.stream.read().decode("utf-8-sig")));rows=[]
                for row in rd:
                    d={str(k).lower().strip():v for k,v in row.items() if k}
                    title=d.get("title",d.get("headline",""));text=d.get("text",d.get("content",d.get("article",d.get("news",""))))
                    if not (title+" "+text).strip():continue
                    r=predict((title+" "+text).strip(),"HYBRID");store(title,text,d.get("source_url",""),r)
                    rows.append({"title":title or text[:80],"category":r["category"],"confidence":round(r["confidence"]*100,2)})
                session["bulk_rows"]=rows;flash(f"Classified {len(rows)} article(s).","success")
            except Exception as e:flash(str(e),"danger")
    return render_template("bulk.html",rows=rows)
@app.route("/bulk/export")
@login_required
def export_bulk():
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=["title","category","confidence"]);w.writeheader();w.writerows(session.get("bulk_rows",[]))
    return send_file(io.BytesIO(out.getvalue().encode("utf-8-sig")),mimetype="text/csv",as_attachment=True,download_name="news_predictions.csv")
@app.route("/history")
@login_required
def history():
    q=request.args.get("q","").strip();cat=request.args.get("category","");saved=request.args.get("saved")=="1"
    sql="SELECT * FROM articles WHERE user_id=?";args=[session["uid"]]
    if q:sql+=" AND (title LIKE ? OR content LIKE ?)";args.extend([f"%{q}%",f"%{q}%"])
    if cat:sql+=" AND category=?";args.append(cat)
    if saved:sql+=" AND saved=1"
    sql+=" ORDER BY id DESC LIMIT 300"
    with conn() as c:rows=c.execute(sql,args).fetchall()
    return render_template("history.html",rows=rows,q=q,selected=cat,saved_only=saved)
@app.route("/article/<int:aid>/save",methods=["POST"])
@login_required
def save(aid):
    with conn() as c:
        r=c.execute("SELECT saved FROM articles WHERE id=? AND user_id=?",(aid,session["uid"])).fetchone()
        if r:c.execute("UPDATE articles SET saved=? WHERE id=?",(0 if r["saved"] else 1,aid))
    return redirect(request.referrer or url_for("history"))
@app.route("/article/<int:aid>/feedback",methods=["POST"])
@login_required
def feedback(aid):
    cat=request.form.get("correct_category","")
    if cat not in CATEGORIES:flash("Select a valid category.","danger")
    else:
        with conn() as c:
            if c.execute("SELECT id FROM articles WHERE id=? AND user_id=?",(aid,session["uid"])).fetchone():
                c.execute("INSERT INTO feedback(article_id,user_id,correct_category,created_at) VALUES(?,?,?,?)",(aid,session["uid"],cat,datetime.now().isoformat(timespec="seconds")))
        flash("Feedback stored. It does not automatically retrain the model.","success")
    return redirect(request.referrer or url_for("history"))
@app.route("/model-lab",methods=["GET","POST"])
@login_required
def model_lab():
    status=None
    if request.method=="POST":
        model=request.form.get("model","HYBRID").upper(); action=request.form.get("action","train")
        f=request.files.get("dataset")
        path=None
        if f and f.filename:
            if not f.filename.lower().endswith(".csv"):status={"error":"Dataset must be CSV."}
            else:
                os.makedirs(os.path.join(ROOT,"data"),exist_ok=True);path=os.path.join(ROOT,"data","uploaded_training.csv");f.save(path)
        if status is None:
            try:
                if action=="evaluate":status=evaluate()
                else:status=train_mlp(path) if model=="MLP" else train_optional(model,path)
            except Exception as e:status={"error":str(e)}
    return render_template("model_lab.html",status=status)
@app.route("/api/classify",methods=["POST"])
@login_required
def api_classify():
    d=request.get_json(silent=True) or {};text=(str(d.get("title",""))+" "+str(d.get("text",""))).strip()
    if len(text)<15:return jsonify({"error":"Provide at least 15 characters."}),400
    try:return jsonify(predict(text,str(d.get("model","HYBRID"))))
    except Exception as e:return jsonify({"error":str(e)}),400
if __name__=="__main__":app.run(host="127.0.0.1",port=5000,debug=True)
