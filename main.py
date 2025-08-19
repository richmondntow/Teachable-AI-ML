from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import joblib, os, uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

models_dir = "user_models"
os.makedirs(models_dir, exist_ok=True)

@app.post("/train/{username}")
async def train_model(username: str, file: UploadFile, task: str = Form(...), target_column: str = Form(None)):
    df = pd.read_csv(file.file)
    if task in ["classification","regression"] and not target_column:
        raise HTTPException(status_code=400, detail="Target column required")
    if task in ["classification","regression"]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_id = str(uuid.uuid4())
    model_path = os.path.join(models_dir, f"{username}_{model_id}.joblib")
    metrics = {}
    if task=="classification":
        model = SGDClassifier(max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, preds)
    elif task=="regression":
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics["mse"] = mean_squared_error(y_test, preds)
    elif task=="clustering":
        model = KMeans(n_clusters=3)
        model.fit(df)
        metrics["silhouette_score"] = silhouette_score(df, model.labels_)
    elif task=="anomaly_detection":
        model = IsolationForest()
        model.fit(df)
        preds = model.predict(df)
        metrics["anomaly_ratio"] = (preds==-1).mean()
    else:
        raise HTTPException(status_code=400, detail="Invalid task")
    joblib.dump(model, model_path)
    return {"message":"Model trained","model_id":model_id,"metrics":metrics}

