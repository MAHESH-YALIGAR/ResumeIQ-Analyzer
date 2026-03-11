import joblib
import io
import uvicorn
import pdfplumber
import nltk
import base64
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from data import skilled_db


# ---------------- LOAD MODEL ----------------

model = joblib.load("logestic_regression.joblib")
vectorizer = joblib.load("vectorizer.joblib")


# ---------------- FASTAPI APP ----------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- NLTK DOWNLOAD ----------------

nltk.download("punkt")
nltk.download("stopwords")


# ---------------- TEXT EXTRACTION ----------------

def extract_text(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text.strip()


# ---------------- TEXT PREPROCESS ----------------

def preprocess(text):

    tokens = word_tokenize(text.lower())

    tokens = [t for t in tokens if t.isalpha()]

    tokens = [t for t in tokens if t not in stopwords.words("english")]

    return " ".join(tokens)


# ---------------- SKILL EXTRACTION ----------------

def extract_skills(text):

    found = []
    all_skills = []

    for category, skills in skilled_db.items():
        all_skills.extend(skills)

    for skill in all_skills:
        if skill.lower() in text:
            found.append(skill)

    return found


# ---------------- CREATE CHART ----------------

def create_chart(role_scores):

    roles = list(role_scores.keys())
    scores = list(role_scores.values())

    fig, ax = plt.subplots(figsize=(8,4))

    bars = ax.barh(roles, scores)

    ax.invert_yaxis()

    ax.set_xlabel("Confidence (%)")
    ax.set_title("Predicted Job Roles")

    ax.bar_label(bars, fmt="%.2f%%")

    buffer = io.BytesIO()

    plt.tight_layout()

    plt.savefig(buffer, format="png")

    plt.close()

    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode()

    return image_base64


# ---------------- API ENDPOINT ----------------

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):

    contents = await file.read()

    # Extract text
    text = extract_text(io.BytesIO(contents))

    # Preprocess text
    clean_text = preprocess(text)

    # Extract skills
    skills = extract_skills(clean_text)

    # Vectorize resume
    vector = vectorizer.transform([clean_text])

    # Predict role
    prediction = model.predict(vector)[0]

    # Get probabilities
    probabilities = model.predict_proba(vector)[0]

    role_scores = {}

    for role, score in zip(model.classes_, probabilities):
        role_scores[role] = round(score * 100, 2)

    # Create matplotlib chart
    chart_image = create_chart(role_scores)

    print("Prediction:", prediction)
    print("Skills:", skills)
    print("Role Scores:", role_scores)

    return {
        "filename": file.filename,
        "predicted_role": prediction,
        "skills_found": skills,
        "role_confidence": role_scores,
        "chart": chart_image
    }


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8081, reload=True)