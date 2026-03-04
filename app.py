import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PDF + Report Generation
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import re
import uuid
import os
import matplotlib.pyplot as plt
from reportlab.platypus import Image

from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from pydantic import BaseModel
from typing import List

class V2AdaptRequest(BaseModel):
    user_id: str
    questions: List[str]

class V2PredictRequest(BaseModel):
    user_id: str
    text: str


# =====================================================
# FASTAPI SETUP
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "pritish0007/deepbloom-v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global placeholders
tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("Loading DeepBloom model from HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
# =====================================================
# REQUEST FORMATS
# =====================================================
class Question(BaseModel):
    text: str


# Multiple question input
class Assessment(BaseModel):
    questions: list[str]


# =====================================================
# LABEL MAP
# =====================================================
label_map = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyse",
    4: "Evaluate",
    5: "Create"
}

# =====================================================
# BLOOM ACTION VERBS (Educational reasoning)
# =====================================================
bloom_verbs = {
    "Remember": ["define", "list", "name", "identify", "recall"],
    "Understand": ["explain", "describe", "summarize", "interpret"],
    "Apply": ["apply", "solve", "use", "demonstrate", "implement"],
    "Analyse": ["analyze", "compare", "differentiate", "examine"],
    "Evaluate": ["evaluate", "justify", "criticize", "assess"],
    "Create": ["design", "create", "develop", "construct", "formulate"]
}
def clean_exam_text(text):

    lines = text.split("\n")
    cleaned = []

    noise_patterns = [
        "sample question paper",
        "general instructions",
        "section",
        "page",
        "marks",
        "time allowed",
        "draw neat figures",
        "wherever required",
        "code no",
        "cbse",
        "visit",
        "www",
        "http"
    ]

    for line in lines:

        l = line.strip()
        lower = l.lower()

        # remove very short fragments
        if len(l) < 12:
            continue

        # remove headers/instructions
        if any(n in lower for n in noise_patterns):
            continue

        # remove option-only lines
        if re.match(r"^\(?[a-dA-D]\)", l):
            continue

        cleaned.append(l)

    return cleaned
# =====================================================
# EXPLANATION GENERATOR
# =====================================================
def generate_explanation(question_text, predicted_level):
    text = question_text.lower()
    explanations = []

    # Verb detection
    for level, verbs in bloom_verbs.items():
        for verb in verbs:
            if verb in text:
                explanations.append(
                    f"Detected cognitive verb '{verb}' associated with {level} level."
                )

    # Educational reasoning
    if predicted_level == "Remember":
        explanations.append("Question focuses on recalling factual information.")

    elif predicted_level == "Understand":
        explanations.append("Question requires conceptual explanation or understanding.")

    elif predicted_level == "Apply":
        explanations.append("Question expects use of knowledge in a practical scenario.")

    elif predicted_level == "Analyse":
        explanations.append("Question requires breaking concepts into parts or reasoning.")

    elif predicted_level == "Evaluate":
        explanations.append("Question asks for judgement or critical assessment.")

    elif predicted_level == "Create":
        explanations.append("Question demands generation of new ideas or design.")

    if not explanations:
        explanations.append("Prediction based on semantic understanding of the question.")

    return explanations


# =====================================================
# SINGLE QUESTION PREDICTION CORE
# =====================================================
def predict_single_question(question_text):

    inputs = tokenizer(
        question_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "level": label_map[idx.item()],
            "confidence": round(prob.item(), 3)
        })

    return results

def predict_level(question_text):

    inputs = tokenizer(
        question_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_index = torch.argmax(probs).item()

    return label_map[pred_index]

# =====================================================
# MULTI QUESTION ANALYSIS
# =====================================================
def analyze_questions(question_list):

    counts = {
        "Remember": 0,
        "Understand": 0,
        "Apply": 0,
        "Analyse": 0,
        "Evaluate": 0,
        "Create": 0
    }

    for q in question_list:
        predictions = predict_single_question(q)
        level = predictions[0]["level"]
        counts[level] += 1

    total = len(question_list)

    percentages = {
        level: round((count / total) * 100, 2)
        for level, count in counts.items()
    }

    return percentages


# =====================================================
# EDUCATIONAL INSIGHT ENGINE
# =====================================================
def generate_assessment_insight(distribution):

    low_order = distribution["Remember"] + distribution["Understand"]
    high_order = (
        distribution["Analyse"]
        + distribution["Evaluate"]
        + distribution["Create"]
    )

    if low_order > 60:
        return "Assessment is dominated by lower-order cognitive questions."

    elif high_order > 40:
        return "Assessment promotes higher-order thinking skills."

    else:
        return "Assessment shows moderate cognitive balance."
# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_text_from_pdf(pdf_file):

    pdf_file.file.seek(0)   # ⭐ IMPORTANT
    pdf_bytes = pdf_file.file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()

    return full_text
# =====================================================
# SMART QUESTION DETECTOR (Research-grade)
# =====================================================
# =====================================================
# SIMPLE STRUCTURED QUESTION DETECTOR
# =====================================================
def extract_questions(text):

    lines = text.split("\n")

    questions = []
    current_question = ""

    # Pattern for detecting question start
    pattern = re.compile(
        r"""
        ^\s*
        (
            \(?\d+\)?[\.\)]?         |   # 1. 1) (1)
            \(?[a-zA-Z]\)?[\.\)]?    |   # a. a) (a)
            \(?[ivxIVX]+\)?[\.\)]?   |   # i. ii. iv.
            [•\-\*\u2022]                # bullet symbols
        )
        \s+
        """,
        re.VERBOSE
    )

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # If line matches question start
        if pattern.match(line):

            if current_question:
                questions.append(current_question.strip())

            current_question = pattern.sub("", line).strip()

        else:
            # continuation of previous question
            if current_question:
                current_question += " " + line

    if current_question:
        questions.append(current_question.strip())

    return questions
# QUESTION SPLITTING
# =====================================================

# =====================================================
# INTERNAL PREDICTION FUNCTION
# =====================================================
def classify_question(question_text):

    inputs = tokenizer(
        question_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_index = torch.argmax(probs).item()

    return label_map[pred_index], round(probs[pred_index].item(), 3)
# =====================================================
# CREATE ANNOTATED PDF
# =====================================================
def create_annotated_pdf(results):

    output_path = "DeepBloom_Report.pdf"

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []

    # ---------------- TITLE ----------------
    story.append(
        Paragraph("DeepBloom Cognitive Assessment Report", styles["Title"])
    )
    story.append(Spacer(1, 20))
    if len(results) == 0:
        results.append({
        "question": "No valid questions detected.",
        "label": "Remember",
        "confidence": 1.0
    })

    # ---------------- DISTRIBUTION ----------------
    counts = {k: 0 for k in label_map.values()}

    for r in results:
        counts[r["label"]] += 1

    total = len(results)

    distribution = {
        level: round((count / total) * 100, 2)
        for level, count in counts.items()
    }

    complexity_score, complexity_level = calculate_complexity_score(distribution)

    # ---------------- SUMMARY ----------------
    summary = f"""
    Total Questions Analysed: {total}<br/>
    Cognitive Complexity Score: {complexity_score}/10<br/>
    Complexity Level: {complexity_level}
    """

    story.append(Paragraph("Statistical Summary", styles["Heading2"]))
    story.append(Paragraph(summary, styles["BodyText"]))
    story.append(Spacer(1, 20))

    # ---------------- CHARTS ----------------
    dist_chart = create_distribution_chart(distribution)
    radar_chart = create_radar_chart(distribution)
    complexity_chart = create_complexity_chart(complexity_score)

    story.append(Paragraph("Cognitive Distribution", styles["Heading2"]))
    story.append(Image(dist_chart, width=400, height=250))

    story.append(Spacer(1, 15))

    story.append(Paragraph("Bloom Radar Analysis", styles["Heading2"]))
    story.append(Image(radar_chart, width=350, height=350))

    story.append(Spacer(1, 15))

    story.append(Paragraph("Complexity Score", styles["Heading2"]))
    story.append(Image(complexity_chart, width=400, height=150))

    story.append(Spacer(1, 20))

    # ---------------- QUESTIONS ----------------
    story.append(
        Paragraph("Question-wise Cognitive Classification", styles["Heading2"])
    )

    for i, item in enumerate(results, 1):

        story.append(
            Paragraph(f"<b>Q{i}:</b> {item['question']}", styles["BodyText"])
        )

        story.append(
            Paragraph(
                f"Bloom Level: {item['label']} "
                f"(Confidence: {item['confidence']})",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 12))

    doc.build(story)

    return output_path

    
# =====================================================
# ROOT CHECK
# =====================================================
@app.get("/")
def home():
    return {"message": "DeepBloom Cognitive Analysis API Running"}


# =====================================================
# SINGLE QUESTION ENDPOINT
# =====================================================
@app.post("/predict")
def predict(data: Question):

    results = predict_single_question(data.text)

    final_level = results[0]["level"]

    explanation = generate_explanation(data.text, final_level)

    return {
        "question": data.text,
        "top_predictions": results,
        "final_prediction": final_level,
        "explanation": explanation
    }


# =====================================================
# COGNITIVE COMPLEXITY SCORE (Research Style)
# =====================================================
def calculate_complexity_score(distribution):

    # research-weighted Bloom hierarchy
    weights = {
        "Remember": 1.0,
        "Understand": 2.0,
        "Apply": 3.0,
        "Analyse": 4.5,
        "Evaluate": 5.5,
        "Create": 6.5
    }

    weighted_sum = 0

    for level, percent in distribution.items():
        weighted_sum += percent * weights[level]

    # maximum possible (if 100% Create)
    max_score = 100 * weights["Create"]

    normalized_score = (weighted_sum / max_score) * 10
    score = round(normalized_score, 2)

    # interpretation layer
    if score < 3:
        complexity_level = "Low cognitive complexity"
    elif score < 6:
        complexity_level = "Moderate cognitive complexity"
    else:
        complexity_level = "High cognitive complexity"

    return score, complexity_level
def create_distribution_chart(distribution):

    levels = list(distribution.keys())
    values = list(distribution.values())

    plt.figure(figsize=(6,4))
    plt.bar(levels, values)
    plt.title("Bloom Cognitive Distribution")
    plt.ylabel("Percentage")

    path = "distribution_chart.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path
import numpy as np

def create_radar_chart(distribution):

    labels = list(distribution.keys())
    values = list(distribution.values())

    values += values[:1]  # close circle

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    path = "radar_chart.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path
def create_complexity_chart(score):

    plt.figure(figsize=(6,2))

    plt.barh(["Complexity"], [score])
    plt.xlim(0,10)
    plt.title(f"Cognitive Complexity Score: {score}/10")

    path = "complexity_chart.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path


# =====================================================
# ASSESSMENT ANALYSIS ENDPOINT
# =====================================================
@app.post("/analyze-assessment")
def analyze_assessment(data: Assessment):

    distribution = analyze_questions(data.questions)

    insight = generate_assessment_insight(distribution)

    complexity_score, complexity_level = calculate_complexity_score(distribution)

    return {
        "total_questions": len(data.questions),
        "cognitive_distribution_percent": distribution,
        "complexity_score_out_of_10": complexity_score,
        "complexity_level": complexity_level,
        "insight": insight
    }

# =====================================================
# PDF UPLOAD & ANALYSIS
# =====================================================
@app.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):

    try:
        # correct usage
        text = extract_text_from_pdf(file)
        cleaned_lines = clean_exam_text(text)
        questions = extract_questions("\n".join(cleaned_lines))

        results = []

        for q in questions:
            label, confidence = classify_question(q)

            results.append({
                "question": q,
                "label": label,
                "confidence": confidence
            })

        output_pdf = create_annotated_pdf(results)

        return FileResponse(
            output_pdf,
            media_type="application/pdf",
            filename="DeepBloom_Report.pdf"
        )

    except Exception as e:
        print("PDF ERROR:", e)
        return {"error": str(e)}
    
# ============================================================
# 🔵 DEEPBLOOM V2 — DOMAIN ADAPTIVE VERSION (RESEARCH GRADE)
# ============================================================

from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ============================================================
# 1️⃣ EMBEDDING EXTRACTION (FOR CLUSTERING)
# ============================================================

def extract_embeddings(text_list):

    embeddings = []

    for text in text_list:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.base_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding[0])

    return np.array(embeddings)


# ============================================================
# 2️⃣ CLUSTER VALIDATION
# ============================================================

def cluster_validation(texts, labels, n_clusters=6):

    embeddings = extract_embeddings(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    validated_indices = []

    for cluster in range(n_clusters):

        cluster_indices = np.where(cluster_ids == cluster)[0]
        cluster_labels = [labels[i] for i in cluster_indices]

        if len(cluster_labels) == 0:
            continue

        dominant = max(set(cluster_labels), key=cluster_labels.count)
        consistency = cluster_labels.count(dominant) / len(cluster_labels)

        if consistency > 0.7:
            validated_indices.extend(cluster_indices.tolist())

    return validated_indices


# ============================================================
# 3️⃣ ADAPT MODEL V2 (CLUSTER + ENTROPY + KL STABILITY)
# ============================================================

def adapt_model_v2(unlabeled_texts, user_id):

    texts = []
    labels = []

    # -------------------------------
    # PSEUDO LABELING (V1)
    # -------------------------------
    for text in unlabeled_texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        confidence, pred = torch.max(probs, dim=0)

        if confidence.item() > 0.85:
            texts.append(text)
            labels.append(pred.item())

    if len(texts) < 15:
        return {"error": "Not enough confident samples for adaptation."}

    # -------------------------------
    # CLUSTER VALIDATION
    # -------------------------------
    validated_indices = cluster_validation(texts, labels)

    if len(validated_indices) < 10:
        return {"error": "Cluster validation removed too many samples."}

    pseudo_data = [
        {"text": texts[i], "label": labels[i]}
        for i in validated_indices
    ]

    # -------------------------------
    # DATASET CREATION
    # -------------------------------
    dataset = Dataset.from_list(pseudo_data)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # -------------------------------
    # LOAD BASE MODEL (V1 FROZEN)
    # -------------------------------
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    base_model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.to(device)

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-5)

    # Frozen V1 for KL stability
    v1_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    v1_model.to(device)
    v1_model.eval()

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    lora_model.train()

    for epoch in range(5):

        for batch in dataset:

            inputs = {
                "input_ids": batch["input_ids"].unsqueeze(0).to(device),
                "attention_mask": batch["attention_mask"].unsqueeze(0).to(device),
                "labels": torch.tensor([batch["label"]]).to(device)
            }

            outputs = lora_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            # Cross-entropy
            ce_loss = outputs.loss

            # Entropy Regularization
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-10),
                dim=1
            ).mean()

            # KL Stability
            with torch.no_grad():
                v1_outputs = v1_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

            v1_probs = torch.softmax(v1_outputs.logits/0.8, dim=1)
            kl_loss = F.kl_div(
                torch.log(probs + 1e-10),
                v1_probs,
                reduction="batchmean"
            )

            loss = ce_loss + 0.01 * entropy + 0.1 * kl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # -------------------------------
    # SAVE USER-SPECIFIC ADAPTER
    # -------------------------------
    save_path = f"user_models/{user_id}"
    os.makedirs(save_path, exist_ok=True)

    lora_model.save_pretrained(save_path)

    return {
    "status": "DeepBloom V2 adaptation complete",
    "total_input_samples": len(unlabeled_texts),
    "pseudo_labeled_samples": len(pseudo_data),
    "validated_samples": len(validated_indices),
    "entropy_regularization": True,
    "kl_divergence_stability": True,
    "adapter_saved": True
}


# ============================================================
# 4️⃣ PREDICT USING V2
# ============================================================

def predict_with_v2(text, user_id):

    adapter_path = f"user_models/{user_id}"

    if not os.path.exists(adapter_path):
        return {"error": "V2 model not found for this user."}

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.to(device)
    lora_model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = lora_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    confidence, pred = torch.max(probs, dim=0)

    return {
        "prediction": label_map[pred.item()],
        "confidence": round(confidence.item(), 3)
    }


# ============================================================
# 5️⃣ V2 API ENDPOINTS
# ============================================================

@app.post("/deepbloom-v2/adapt")
def adapt_v2(request: V2AdaptRequest):

    result = adapt_model_v2(
        request.questions,
        request.user_id
    )

    return result


@app.post("/deepbloom-v2/predict")
def predict_v2(request: V2PredictRequest):

    result = predict_with_v2(
        request.text,
        request.user_id
    )


    return result



import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

