

import joblib
import io
import uvicorn
import pdfplumber
import nltk
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from  data  import skilled_db

# Load ML model and vectorizer
model = joblib.load("logestic_regression.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data (run once)
nltk.download("punkt")
nltk.download("stopwords")


# ---------- TEXT EXTRACTION ----------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


# ---------- PREPROCESS TEXT ----------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)


# ---------- SKILL EXTRACTION ----------
def extract_skills(text):
    found = []
    all_skills = []

    for category, skills in skilled_db.items():
        all_skills.extend(skills)

    for skill in all_skills:
        if skill.lower() in text:
            found.append(skill)

    return found


# ---------- API ENDPOINT ----------
@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...)
):

    contents = await file.read()

    # Extract text from PDF
    text = extract_text(io.BytesIO(contents))

    # Clean text
    clean_text = preprocess(text)

    # Extract skills
    skills = extract_skills(clean_text)

    # Vectorize text
    vector = vectorizer.transform([clean_text])

    # Predict role
    prediction = model.predict(vector)[0]

    # Confidence scores
    probabilities = model.predict_proba(vector)[0]

    role_scores = {}
    for role, score in zip(model.classes_, probabilities):
        role_scores[role] = round(score * 100, 2)

    print(f"the final result is the {prediction} and {skills}  and {role_scores}")
    return {
        "filename": file.filename,
        "predicted_role": prediction,
        "skills_found": skills,
        "role_confidence": role_scores
    }

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8081, reload=True)

