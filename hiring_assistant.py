import os
import re
import json
import time
import uuid
import traceback
from html import escape as html_escape

import streamlit as st
from dotenv import load_dotenv

from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from pymongo.mongo_client import MongoClient

load_dotenv()

EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")

POSITIVE_KEYWORDS = {"yes", "ready", "ok", "okay", "sure", "confident", "yep", "yeah", "lets go", "let's go", "go ahead"}
NEGATIVE_KEYWORDS = {"no", "not ready", "unprepared", "not prepared", "nope", "later", "can't", "cannot"}
GREETING_KEYWORDS = {"hi", "hello", "hey", "hey there", "greetings"}
SKIP_KEYWORDS = {"i don't know", "idk", "dont know", "don't know", "skip", "pass", "next"}

# Header of the page
st.set_page_config(page_title="TalentScout Hiring", page_icon="🤖", layout="wide")

# CSS
CSS = f"""
<style>
/* Reduce Streamlit top padding so chat sits right under header */
.block-container {{ padding-top: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }}
.section-wrapper, main > div[role="main"] > div:nth-child(1) {{ padding-top: 0rem !important; }}

:root {{
  --bg: #0b1220;
  --muted: #94a3b8;
}}
body {{ background: var(--bg); color: #e6eefc; }}

.header {{
  display:flex; align-items:center; justify-content:space-between;
  padding:40px 18px; border-bottom:1px solid rgba(255,255,255,0.03); margin-bottom:0;
}}
.header-left {{display:flex; align-items:center; gap:12px}}
.logo-emoji {{ font-size:28px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:linear-gradient(135deg,#7c3aed,#4f46e5); color:white; }}
.header-right {{display:flex; gap:14px; align-items:center}}
.badge {{background:rgba(255,255,255,0.04); padding:6px 10px; border-radius:18px; color:#f8fafc; font-size:13px}}
.status-dot {{width:10px; height:10px; border-radius:50%; background:#34d399; box-shadow:0 0 8px rgba(52,211,153,0.18)}}

.container {{ display:flex; gap:12px; padding:8px 16px 18px 16px; margin-top:0; }}
.chat-col {{ flex:1; min-height:68vh; }}
.side-col {{ width:360px; }}

.message-row {{ display:flex; gap:12px; margin:10px 0; align-items:flex-start; }}
.message-row.assistant {{ justify-content:flex-start; }}
.message-row.user {{ justify-content:flex-end; }}

.avatar-emoji {{
  width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:20px; flex:0 0 44px; box-shadow: 0 6px 18px rgba(2,6,23,0.6);
  border:1px solid rgba(255,255,255,0.03);
}}
.avatar-emoji.assistant {{ background: linear-gradient(135deg,#7c3aed,#4f46e5); }}
.avatar-emoji.user {{ background: linear-gradient(135deg,#06b6d4,#06b6d4); }}

.bubble {{ padding:12px 16px; border-radius:14px; max-width:68%; line-height:1.35; font-size:15px; position:relative; }}
.bubble.assistant {{
  background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.04);
  box-shadow: 0 6px 24px rgba(2,6,23,0.6); border-top-left-radius:8px;
}}
.bubble.user {{
  background: linear-gradient(135deg,#7c3aed,#06b6d4); color:white;
  border-top-right-radius:8px; box-shadow: 0 6px 24px rgba(9,10,27,0.5);
  display:inline-block;
}}

.timestamp {{ display:block; font-size:12px; color:var(--muted); margin-top:8px; }}

.session-drawer {{
  background:#071025; border-radius:12px; padding:20px; color:#e6eefc;
}}
.progress-bar {{ height:10px; background:rgba(255,255,255,0.06); border-radius:8px; overflow:hidden }}
.progress-inner {{ height:100%; width:50%; background:linear-gradient(90deg,#7c3aed,#06b6d4) }}
.small-muted {{ color:var(--muted); font-size:12px }}
.kv {{ display:flex; justify-content:space-between; margin-top:10px }}
.kv .k {{ color:var(--muted); font-size:13px }} .kv .v {{ font-weight:600; font-size:14px }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Utility functions
def extract_json_from_text(text: str):
    if not text:
        return None
    m_obj = re.search(r"\{.*\}", text, re.DOTALL)
    if m_obj:
        try:
            return json.loads(m_obj.group())
        except Exception:
            pass
    m_arr = re.search(r"\[.*\]", text, re.DOTALL)
    if m_arr:
        try:
            return json.loads(m_arr.group())
        except Exception:
            pass
    return None

class CandidateInfoParser(BaseOutputParser):
    def parse(self, text: str):
        parsed = extract_json_from_text(text)
        return parsed if isinstance(parsed, dict) else {}

def extract_email_from_text(text: str) -> str:
    if not text:
        return ""
    m = EMAIL_REGEX.search(text)
    return m.group(1).strip() if m else ""

def extract_phone_from_text(text: str) -> str:
    if not text:
        return ""
    m = PHONE_REGEX.search(text)
    if not m:
        return ""
    ph = m.group(1)
    ph_norm = re.sub(r"[^\d+]", "", ph)
    return ph_norm

def extract_name_from_text(text: str) -> str:
    if not text:
        return ""
    if ":" in text:
        parts = text.split(":", 1)
        cand = parts[1].strip()
        if EMAIL_REGEX.search(cand) or PHONE_REGEX.search(cand):
            return ""
        return cand
    t = re.sub(r"^(my name is|i am|name is)\s*", "", text.strip(), flags=re.IGNORECASE)
    if EMAIL_REGEX.search(t) or PHONE_REGEX.search(t):
        return ""
    return t.strip()

# ----------------- Mongo helpers -----------------
def get_mongo_collections():
    if st.session_state.get("_mongo_init_done"):
        return st.session_state.get("_personal_col"), st.session_state.get("_qa_col"), st.session_state.get("_mongo_status")
    status = {"ok": False, "msg": "Not attempted"}
    personal_col = None
    qa_col = None
    full_uri = os.getenv("MONGODB_URI")
    if full_uri:
        uri = full_uri
    else:
        password = os.getenv("DB_PASSWORD")
        if not password:
            status["msg"] = "MONGODB_URI or DB_PASSWORD not set. DB operations disabled."
            st.session_state["_mongo_status"] = status
            st.session_state["_mongo_init_done"] = True
            return None, None, status
        uri = f"mongodb+srv://espi3088:{password}@espi.bebf5dd.mongodb.net/?retryWrites=true&w=majority&appName=Espi"
    try:
        client = st.session_state.get("_mongo_client")
        if client is None:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            st.session_state["_mongo_client"] = client
        client.admin.command("ping")
        db = client["mytestdb"]
        personal_col = db["personal_details"]
        qa_col = db["qa_round"]
        status["ok"] = True
        status["msg"] = "Connected"
    except Exception as e:
        st.session_state["_mongo_last_error_trace"] = traceback.format_exc()
        status["ok"] = False
        status["msg"] = f"MongoDB connection failed: {e}"
    st.session_state["_mongo_init_done"] = True
    st.session_state["_personal_col"] = personal_col
    st.session_state["_qa_col"] = qa_col
    st.session_state["_mongo_status"] = status
    return personal_col, qa_col, status

def save_results_to_db(candidate: dict) -> dict:
    result = {"ok": False, "personal_id": None, "qa_id": None, "errors": [], "tracebacks": {}}
    personal_col, qa_col, mongo_status = get_mongo_collections()
    if not mongo_status or not mongo_status.get("ok"):
        result["errors"].append("MongoDB not configured or connection failed: " + (mongo_status.get("msg") if mongo_status else "no status"))
        return result
    personal_doc = {
        "session_id": candidate.get("session_id"),
        "full_name": candidate.get("full_name"),
        "email": candidate.get("email"),
        "phone": candidate.get("phone"),
        "years_of_experience": candidate.get("years_of_experience"),
        "desired_position": candidate.get("desired_position"),
        "current_location": candidate.get("current_location"),
        "tech_stack_structured": candidate.get("tech_stack_structured"),
        "tech_stack_flat": candidate.get("tech_stack"),
        "created_at": time.time()
    }
    try:
        res = personal_col.insert_one(personal_doc)
        result["personal_id"] = res.inserted_id
    except Exception:
        result["tracebacks"]["personal_insert_first"] = traceback.format_exc()
        try:
            res = personal_col.insert_one(personal_doc)
            result["personal_id"] = res.inserted_id
        except Exception as e2:
            result["tracebacks"]["personal_insert_second"] = traceback.format_exc()
            result["errors"].append(f"Failed to save personal details: {e2}")
            return result
    total_scores = [s.get("score") for s in candidate.get("scores", []) if s.get("score") is not None]
    qa_doc = {
        "session_id": candidate.get("session_id"),
        "personal_id": result["personal_id"],
        "desired_position": candidate.get("desired_position"),
        "technical_questions": candidate.get("technical_questions"),
        "technical_answers": candidate.get("technical_answers"),
        "scores": candidate.get("scores"),
        "total_score": sum(total_scores) if total_scores else None,
        "created_at": time.time()
    }
    try:
        res = qa_col.insert_one(qa_doc)
        result["qa_id"] = res.inserted_id
    except Exception:
        result["tracebacks"]["qa_insert_first"] = traceback.format_exc()
        try:
            res = qa_col.insert_one(qa_doc)
            result["qa_id"] = res.inserted_id
        except Exception as e2:
            result["tracebacks"]["qa_insert_second"] = traceback.format_exc()
            result["errors"].append(f"Failed to save QA doc: {e2}")
    result["ok"] = bool(result["personal_id"])
    return result

# ----------------- LLM init & wrappers -----------------
def init_llms():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, None
    try:
        llm_main = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
        llm_eval = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key)
        return llm_main, llm_eval
    except Exception:
        return None, None

def llm_intent_classify(llm, user_text: str) -> str:
    prompt = (
        'You are an intent classifier for a hiring assistant. Classify into Positive/Negative/Neutral. '
        'Return JSON like {"intent":"Positive"}.\n'
        f'Text: {user_text}'
    )
    try:
        resp = llm.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            intent = str(j.get("intent", "")).strip().capitalize()
            if intent in {"Positive", "Negative", "Neutral"}:
                return intent
    except Exception:
        pass
    return "Neutral"

def llm_check_relevance(llm, question: str, answer: str) -> dict:
    prompt = (
        f"You are a concise assistant that decides whether a user's SHORT answer is relevant to the asked question.\n"
        f"Actually for a Hiring Process of an organisation, the interviewer asks a question and the user answers the question\n"
        f"Being an assistant you just need to checkk if the user answer is relevant to the question or not\n"
        f"If question is about phone number, check if the format of phone number is correct or not (Like ph no. having ten digits or not\n)"
        f"If question is about email, check if the format of the email address is correct or not\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Respond with a JSON object exactly like:\n"
        "{\"relevant\": true_or_false, \"explanation\": \"one-sentence reason why it's (ir)relevant\"}\n"
        "Be concise and factual."
    )
    try:
        resp = llm.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            return {"relevant": bool(j.get("relevant", False)), "explanation": str(j.get("explanation", "")).strip()}
    except Exception:
        pass
    return {"relevant": True, "explanation": "Assuming relevant (parse failed)."}

def llm_check_location(llm, answer: str) -> dict:
    prompt = (
        "You are an assistant that checks if a short user reply is a plausible EARTH location (city, state/province, or country).\n"
        "Only accept real-world locations on Earth. Reject planets, moons, fictional places, or vague nonsense.\n"
        "Answer with a strict JSON object exactly like:\n"
        "{\"valid\": true_or_false, \"explanation\": \"one-sentence reason\"}\n\n"
        f"User reply: {answer}\n"
    )
    try:
        resp = llm.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            return {"valid": bool(j.get("valid", False)), "explanation": str(j.get("explanation", "")).strip()}
    except Exception:
        pass
    return {"valid": False, "explanation": "LLM parse failed; treat as invalid."}

def llm_extract_tech_stack(llm, raw_text: str) -> dict:
    prompt = (
        "You are an expert in identifying technology stacks. "
        "Extract all mentioned technologies from the given text and classify them into the following categories:\n"
        "- languages: programming languages (e.g., Python, Java, C++, JavaScript, Go).\n"
        "- frameworks: libraries or frameworks (e.g., React, Django, Flask, Spring, Angular).\n"
        "- databases: SQL/NoSQL or data storage systems (e.g., MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch).\n"
        "- tools: software, platforms, cloud services, or development tools (e.g., Git, Docker, Kubernetes, AWS, Jenkins).\n"
        "- other: anything technical that doesn’t fit the above categories.\n\n"
        "Rules:\n"
        "1. Only include items explicitly or clearly implied in the text.\n"
        "2. Each item should appear only once in the appropriate category.\n"
        "3. Maintain consistent casing (e.g., 'Python' not 'python').\n"
        "4. If nothing fits a category, return an empty list for it.\n"
        "5. Return output strictly as a JSON object with this format:\n"
        '{"languages": [], "frameworks": [], "databases": [], "tools": [], "other": []}\n\n'
        f"Text: {raw_text}\n"
        "JSON Output:"
    )
    try:
        resp = llm.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            return {k: [str(x).strip() for x in j.get(k, [])] for k in ("languages", "frameworks", "databases", "tools", "other")}
    except Exception:
        pass
    return {"languages": [], "frameworks": [], "databases": [], "tools": [], "other": []}

def generate_technical_questions(llm, tech_stack_list: list, desired_position: str, n_questions: int = 5) -> list:
    if not tech_stack_list:
        return []
    prompt = (
        "You are a professional technical interviewer. "
        "Your main priority is the DESIRED POSITION when creating questions. "
        "Use the listed technologies only as supporting context. "
        "Generate short, logical, and reasoning-based technical questions. "
        "Do NOT ask full coding problems, long case studies, or overly descriptive scenarios. "
        "Each question must:\n"
        "- Be tailored to the desired position first.\n"
        "- Be related to the listed technologies when relevant.\n"
        "- Be concise (6–12 words).\n"
        "- Be in a numbered list (one per line).\n\n"
        f"Desired Position: {desired_position}\n"
        f"Technologies: {', '.join(tech_stack_list)}\n"
        f"Generate {n_questions} questions:\n"
    )
    try:
        resp = llm.invoke(prompt)
        lines = [ln.strip() for ln in resp.content.strip().splitlines() if ln.strip()]
        qs = [re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", ln) for ln in lines]
        return qs[:n_questions]
    except Exception:
        return []

def llm_detect_synthetic_answer(llm, answer: str) -> dict:
    prompt = (
        "You are an expert in detecting AI-generated text. "
        "Analyze the following short answer and determine if it is synthetic (likely produced by a language model) "
        "or human-written. Consider features like unnatural fluency, generic phrasing, lack of personal detail, "
        "overly polished structure, or repetition.\n\n"
        "Return output strictly in JSON with the following format:\n"
        '{"synthetic": true_or_false, "confidence": 0.0_to_1.0, "explanation": "one clear sentence"}\n\n'
        "Examples:\n"
        "Answer: 'The industrial revolution was a period of great change.'\n"
        "JSON Output: {\"synthetic\": true, \"confidence\": 0.82, \"explanation\": \"The phrasing is generic and polished without human nuance.\"}\n\n"
        "Answer: 'I struggled with Python dictionaries when I started, but practice helped.'\n"
        "JSON Output: {\"synthetic\": false, \"confidence\": 0.77, \"explanation\": \"The response contains personal detail unlikely in synthetic text.\"}\n\n"
        f"Answer: {answer}\n"
        "JSON Output:"
    )

    try:
        resp = llm.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            return {"synthetic": bool(j.get("synthetic", False)), "confidence": float(j.get("confidence", 0.0)), "explanation": str(j.get("explanation", "")).strip()}
    except Exception:
        pass
    return {"synthetic": False, "confidence": 0.0, "explanation": "Detection failed; assume not synthetic."}

def evaluate_answer_with_evaluator(llm_eval, llm_main, question: str, answer: str, desired_position: str, followup_context: bool = False) -> dict:
    followup_instruction = (
        "Do NOT request another follow-up if followup_context True."
        if followup_context
        else "Only suggest a follow-up if it adds value."
    )
    prompt = (
        "You are an objective technical interviewer tasked with evaluating a candidate's short answer. "
        "Your evaluation must focus on the relevance, correctness, clarity, and appropriateness for the desired position.\n\n"

        "Scoring Rules (0–10 scale):\n"
        "- 0–3: Irrelevant, incorrect, or incoherent answer.\n"
        "- 4–6: Partially correct but incomplete or vague.\n"
        "- 7–8: Mostly correct, clear, and relevant to the position.\n"
        "- 9–10: Excellent, precise, well-structured, and position-appropriate.\n\n"

        "Follow-up Rule: Suggest a follow-up only if the answer is unclear, incomplete, or leaves room for elaboration. "
        "If the answer is already strong and sufficient, return an empty string.\n\n"

        f"Desired Position: {desired_position}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"{followup_instruction}\n\n"

        "Return output strictly as JSON in the format:\n"
        "{"
        "\"score\": integer (0–10), "
        "\"justification\": \"one concise sentence\", "
        "\"follow_up\": \"short clarifying question or empty string\""
        "}\n\n"

        "Examples:\n"
        "Answer: 'Python is a programming language.'\n"
        "JSON Output: {\"score\": 5, \"justification\": \"The answer is correct but too basic and incomplete.\", \"follow_up\": \"Can you explain Python's main use cases?\"}\n\n"

        "Answer: 'I optimized queries in PostgreSQL to reduce load times.'\n"
        "JSON Output: {\"score\": 9, \"justification\": \"The answer is specific, relevant, and aligned with database expertise.\", \"follow_up\": \"\"}\n\n"

        "Now evaluate the given answer."

    )
    try:
        resp = llm_eval.invoke(prompt)
        j = extract_json_from_text(resp.content.strip())
        if isinstance(j, dict):
            score = int(j.get("score", 0)) if j.get("score") is not None else 0
            justification = str(j.get("justification", "")).strip()
            follow_up = str(j.get("follow_up", "")).strip()
        else:
            score = min(10, max(0, len(answer.split()) // 3))
            justification = "Auto-scored fallback"
            follow_up = ""
    except Exception:
        score = min(10, max(0, len(answer.split()) // 3))
        justification = "Auto-scored fallback"
        follow_up = ""
    synth = llm_detect_synthetic_answer(llm_main, answer)
    synthetic = bool(synth.get("synthetic", False))
    confidence = float(synth.get("confidence", 0.0))
    synthetic_penalty = 0
    if synthetic and confidence >= 0.6:
        synthetic_penalty = min(score, int(round(2 * confidence)))
        score = max(0, score - synthetic_penalty)
    return {
        "score": score,
        "justification": justification,
        "follow_up": follow_up,
        "synthetic": synthetic,
        "synthetic_penalty": synthetic_penalty,
        "synthetic_confidence": confidence,
        "synthetic_explanation": synth.get("explanation", "")
    }

# App state
def default_candidate() -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "full_name": "",
        "email": "",
        "phone": "",
        "years_of_experience": "",
        "desired_position": "",
        "current_location": "",
        "tech_stack_structured": {"languages": [], "frameworks": [], "databases": [], "tools": [], "other": []},
        "tech_stack": [],
        "technical_questions": [],
        "technical_answers": [],
        "scores": []
    }

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "state" not in st.session_state:
    st.session_state.state = "greetings"
if "candidate" not in st.session_state:
    st.session_state.candidate = default_candidate()
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False
if "expecting_follow_up" not in st.session_state:
    st.session_state.expecting_follow_up = False
if "follow_up_asked_for_current_question" not in st.session_state:
    st.session_state.follow_up_asked_for_current_question = False
if "last_question_for_follow_up" not in st.session_state:
    st.session_state.last_question_for_follow_up = ""
if "expecting_tech_followup" not in st.session_state:
    st.session_state.expecting_tech_followup = False
if "tech_missing_fields" not in st.session_state:
    st.session_state.tech_missing_fields = []
if "placeholder_index" not in st.session_state:
    st.session_state.placeholder_index = None
if "_show_drawer" not in st.session_state:
    st.session_state["_show_drawer"] = False

# Header of the page
st.markdown(f"""
<div class="header">
  <div class="header-left">
    <div class="logo-emoji">🤖</div>
    <div>
      <div style="font-weight:800; font-size:18px">TalentScout</div>
      <div style="color:#9aa4b8; font-size:12px">AI Hiring Assistant</div>
    </div>
  </div>
  <div class="header-right">
    <div class="badge">Personal Info</div>
    <div class="status-dot" title="Session active"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Initial message display
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm the TalentScout Hiring Assistant. I will guide you through a brief hiring screening. Are you ready to begin the hiring process? (yes / no / describe)",
        "ts": time.time()
    })
# Rendeing user chats
def render_chat_and_drawer():
    personal_col, qa_col, mongo_status = get_mongo_collections()
    col1, col2 = st.columns([3, 0.9])
    with col1:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            ts = msg.get("ts", None)
            thinking = msg.get("thinking", False)
            safe = html_escape(content).replace("\n", "<br>")
            ts_str = time.strftime("%I:%M:%S %p", time.localtime(ts)) if ts else ""
            if role == "assistant":
                avatar_html = '<div class="avatar-emoji assistant">🤖</div>'
                bubble_inner = f"<span class='small-muted thinking'>{safe}</span>" if thinking else safe
                bubble_html = f'<div class="bubble assistant">{bubble_inner}<div class="timestamp">{ts_str}</div></div>'
                row_html = f'<div class="message-row assistant">{avatar_html}{bubble_html}</div>'
            else:
                avatar_html = '<div class="avatar-emoji user">👤</div>'
                bubble_html = f'<div class="bubble user">{safe}<div class="timestamp" style="color:rgba(255,255,255,0.9);">{ts_str}</div></div>'
                row_html = f'<div class="message-row user">{bubble_html}{avatar_html}</div>'
            st.markdown(row_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        
        if st.button("Session Info"):
            st.session_state["_show_drawer"] = not st.session_state.get("_show_drawer", False)

        if st.session_state.get("_show_drawer"):
            
            personal = st.session_state.candidate
            filled = sum(1 for k in ("full_name","email","phone","years_of_experience","desired_position","current_location","tech_stack") if personal.get(k))
            progress = int((filled / 7) * 100)

            
            with st.container():
                st.markdown('<div class="session-drawer">', unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'><div style='font-weight:700; font-size:18px'>Session Info</div><div style='font-size:12px; color:#9aa4b8'>Active</div></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top:16px'><div class='small-muted'>Progress</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='progress-bar' style='margin-top:8px'><div class='progress-inner' style='width:{progress}%'></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px; font-weight:700'>{progress}% Complete</div></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top:16px; display:flex; gap:10px; align-items:center;'><div class='avatar-emoji assistant' style='width:42px;height:42px;border-radius:8px; font-size:18px'>🤖</div><div style='font-weight:600'>TalentScout</div></div>", unsafe_allow_html=True)

                
                st.markdown(f"<div class='kv'><div class='k'>Candidate</div><div class='v'>{html_escape(personal.get('full_name','-'))}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><div class='k'>Email</div><div class='v'>{html_escape(personal.get('email','-'))}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><div class='k'>Phone</div><div class='v'>{html_escape(personal.get('phone','-'))}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:18px' class='small-muted'>QA Summary</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px' class='small-muted'>Questions: {len(personal.get('technical_questions',[]))} • Answered: {len([a for a in personal.get('technical_answers',[]) if a])}</div>", unsafe_allow_html=True)

               
                if st.button("Close", key="close_drawer"):
                    st.session_state["_show_drawer"] = False

                st.markdown('</div>', unsafe_allow_html=True)

render_chat_and_drawer()

if not st.session_state.conversation_ended:
    user_text = st.chat_input("Type your message here...")
    if user_text:
        
        st.session_state.messages.append({"role": "user", "content": user_text, "ts": time.time()})
        placeholder = {"role": "assistant", "content": "Thinking...", "thinking": True, "ts": time.time()}
        st.session_state.messages.append(placeholder)
        st.session_state.placeholder_index = len(st.session_state.messages) - 1
        st.session_state.pending = {"text": user_text}
        st.experimental_rerun()

if st.session_state.pending and not st.session_state.conversation_ended:
    pending_text = st.session_state.pending.get("text")
    llm_main, llm_eval = init_llms()
    if not llm_main or not llm_eval:
        idx = st.session_state.placeholder_index
        if idx is not None and 0 <= idx < len(st.session_state.messages):
            st.session_state.messages[idx] = {"role": "assistant", "content": "LLM not configured (set GOOGLE_API_KEY).", "ts": time.time()}
        st.session_state.pending = None
        st.experimental_rerun()

    def replace_placeholder(new_content: str, thinking_flag: bool = False):
        idx = st.session_state.placeholder_index
        if idx is not None and 0 <= idx < len(st.session_state.messages):
            msg = {"role": "assistant", "content": new_content, "ts": time.time()}
            if thinking_flag:
                msg["thinking"] = True
            st.session_state.messages[idx] = msg
            st.session_state.placeholder_index = None
        else:
            st.session_state.messages.append({"role": "assistant", "content": new_content, "ts": time.time()})

    def process_pending(text: str):
        lower = text.strip().lower()
        state = st.session_state.state
        cand = st.session_state.candidate

        # GREETINGS
        if state == "greetings":
            if any(pk in lower for pk in POSITIVE_KEYWORDS):
                st.session_state.state = "personal_details"
                replace_placeholder("Great! Let's start with your personal details. What is your full name?")
            elif any(nk in lower for nk in NEGATIVE_KEYWORDS):
                st.session_state.state = "exit"
                replace_placeholder("No problem. Come back when you're ready. Goodbye!")
                st.session_state.conversation_ended = True
            elif any(gk in lower for gk in GREETING_KEYWORDS) or "describe" in lower or "process" in lower:
                replace_placeholder("I will collect your personal details, ask short technical questions tailored to your desired position and tech stack, and save results. Are you ready to begin? (yes / no)")
            else:
                intent = llm_intent_classify(llm_main, text)
                if intent == "Positive":
                    st.session_state.state = "personal_details"
                    replace_placeholder("Great! Let's start with your personal details. What is your full name?")
                elif intent == "Negative":
                    st.session_state.state = "exit"
                    replace_placeholder("Okay — we will stop here. Goodbye!")
                    st.session_state.conversation_ended = True
                else:
                    replace_placeholder("Please reply 'yes' to begin or 'no' to exit.")

        # PERSONAL DETAILS
        elif state == "personal_details":
            if not cand["full_name"]:
                q = "What is your full name?"
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please provide your full name.")
                else:
                    name_val = extract_name_from_text(text)
                    if not name_val or any(ch.isdigit() for ch in name_val):
                        replace_placeholder("Please provide a valid full name (no digits).")
                    else:
                        cand["full_name"] = name_val
                        replace_placeholder("Thanks. What is your email address?")
            elif not cand["email"]:
                q = "What is your email address?"
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please provide your email address.")
                else:
                    email_found = extract_email_from_text(text)
                    if not email_found:
                        replace_placeholder("That doesn't look like a valid email. Please enter a correct email (ex: name@example.com).")
                    else:
                        cand["email"] = email_found
                        replace_placeholder("Got it. What is your phone number? (include country code if applicable)")
            elif not cand["phone"]:
                q = "What is your phone number?"
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please provide your phone number.")
                else:
                    ph = extract_phone_from_text(text)
                    if not ph:
                        replace_placeholder("That looks invalid. Provide digits only with optional leading + (7-15 digits).")
                    else:
                        cand["phone"] = ph
                        replace_placeholder("How many years of experience do you have? (e.g., 3, 4.5, 10+)")
            elif not cand["years_of_experience"]:
                q = "How many years of experience?"
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please provide years of experience.")
                else:
                    if not re.search(r"\d", text):
                        replace_placeholder("Please include a number, e.g., 3, 4.5, 10+.")
                    else:
                        cand["years_of_experience"] = text.strip()
                        replace_placeholder("What is your desired position? (e.g., Backend Developer)")
            elif not cand["desired_position"]:
                q = "Desired position?"
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please state desired position.")
                else:
                    desired = text.split(":", 1)[1].strip() if ":" in text else text.strip()
                    cand["desired_position"] = desired
                    replace_placeholder("What is your current location (city, state/country)?")
            elif not cand["current_location"]:
                loc = llm_check_location(llm_main, text)
                if not loc["valid"]:
                    replace_placeholder("Invalid location. " + loc["explanation"] + " Please provide a real Earth location.")
                else:
                    cand["current_location"] = text.strip()
                    replace_placeholder("Finally, list your tech stack (languages, frameworks, databases, tools). Free-form is fine.")
            elif not cand["tech_stack"] and not st.session_state.expecting_tech_followup:
                q = "List your tech stack."
                rel = llm_check_relevance(llm_main, q, text)
                if not rel["relevant"]:
                    replace_placeholder("That seems irrelevant. " + rel["explanation"] + " Please provide your tech stack.")
                else:
                    structured = llm_extract_tech_stack(llm_main, text)
                    cand["tech_stack_structured"] = structured
                    missing = []
                    if not structured.get("languages"):
                        missing.append("languages")
                    if not (structured.get("frameworks") or structured.get("databases") or structured.get("tools")):
                        missing.extend([m for m in ("frameworks","databases","tools") if m not in missing])
                    if missing:
                        st.session_state.expecting_tech_followup = True
                        st.session_state.tech_missing_fields = missing.copy()
                        replace_placeholder(f"I couldn't detect your {st.session_state.tech_missing_fields[0]} clearly. Please list them (comma-separated).")
                    else:
                        flat = []
                        for k in ("languages","frameworks","databases","tools","other"):
                            flat.extend([t for t in structured.get(k,[]) if t])
                        cand["tech_stack"] = flat
                        replace_placeholder("Thanks — captured. Proceed to technical questions now? (yes / no)")
            elif st.session_state.expecting_tech_followup:
                if st.session_state.tech_missing_fields:
                    field = st.session_state.tech_missing_fields.pop(0)
                    try:
                        resp = llm_main.invoke(f'Extract a JSON array of {field} from: "{text}"\nReturn only JSON array like ["x","y"]')
                        arr = extract_json_from_text(resp.content.strip())
                        if isinstance(arr, list):
                            cand["tech_stack_structured"].setdefault(field, [])
                            for item in arr:
                                item_s = str(item).strip()
                                if item_s and item_s not in cand["tech_stack_structured"][field]:
                                    cand["tech_stack_structured"][field].append(item_s)
                        else:
                            raise Exception("No array")
                    except Exception:
                        cand["tech_stack_structured"].setdefault(field, [])
                        for item in [p.strip() for p in re.split(r",|/| and ", text) if p.strip()]:
                            if item not in cand["tech_stack_structured"][field]:
                                cand["tech_stack_structured"][field].append(item)
                    if st.session_state.tech_missing_fields:
                        replace_placeholder(f"Thanks. Please also list your {st.session_state.tech_missing_fields[0]}.")
                    else:
                        flat = []
                        for k in ("languages","frameworks","databases","tools","other"):
                            flat.extend([t for t in cand["tech_stack_structured"].get(k,[]) if t])
                        cand["tech_stack"] = flat
                        st.session_state.expecting_tech_followup = False
                        replace_placeholder("Thanks — I've captured your tech stack. Proceed to technical questions now? (yes / no)")
                else:
                    st.session_state.expecting_tech_followup = False
            else:
                if any(pk in lower for pk in POSITIVE_KEYWORDS):
                    st.session_state.state = "question_generation"
                    replace_placeholder("Proceeding to technical questions...")
                    cand["technical_questions"] = generate_technical_questions(llm_main, cand["tech_stack"], cand.get("desired_position","General"), n_questions=5)
                    st.session_state.current_question_index = 0
                    if cand["technical_questions"]:
                        st.session_state.messages.append({"role":"assistant","content":f"Question 1: {cand['technical_questions'][0]}","ts":time.time()})
                    else:
                        st.session_state.messages.append({"role":"assistant","content":"Couldn't generate technical questions. Re-enter tech stack.","ts":time.time()})
                elif any(nk in lower for nk in NEGATIVE_KEYWORDS):
                    st.session_state.state = "exit"
                    replace_placeholder("Okay, stopping here. Goodbye!")
                    st.session_state.conversation_ended = True
                else:
                    replace_placeholder("Reply 'yes' to start questions or 'no' to exit.")

        # QUESTION_GENERATION
        elif state == "question_generation":
            cand = st.session_state.candidate
            if st.session_state.expecting_follow_up:
                if any(skip in lower for skip in SKIP_KEYWORDS):
                    cand["scores"].append({"question": st.session_state.last_question_for_follow_up, "score": None, "justification":"Skipped","follow_up":True})
                    st.session_state.expecting_follow_up = False
                    st.session_state.follow_up_asked_for_current_question = False
                    st.session_state.last_question_for_follow_up = ""
                    st.session_state.current_question_index += 1
                    if st.session_state.current_question_index < len(cand["technical_questions"]):
                        replace_placeholder(f"Question {st.session_state.current_question_index+1}: {cand['technical_questions'][st.session_state.current_question_index]}")
                    else:
                        replace_placeholder("That completes the technical questions. Saving your results now.")
                        st.session_state.state = "exit"
                else:
                    q_text = st.session_state.last_question_for_follow_up
                    rel = llm_check_relevance(llm_main, q_text + " (follow-up)", text)
                    if not rel["relevant"]:
                        replace_placeholder("That follow-up reply seems irrelevant. " + rel["explanation"] + " You may type 'skip' to move on.")
                    else:
                        eval_res = evaluate_answer_with_evaluator(llm_eval, llm_main, q_text, text, cand.get("desired_position","General"), followup_context=True)
                        cand["scores"].append({"question": q_text, "score": eval_res.get("score"), "justification": eval_res.get("justification"), "follow_up": True, "synthetic_penalty": eval_res.get("synthetic_penalty",0)})
                        if cand["technical_answers"]:
                            cand["technical_answers"][-1] += " || Follow-up: " + text.strip()
                        else:
                            cand["technical_answers"].append(text.strip())
                        st.session_state.expecting_follow_up = False
                        st.session_state.follow_up_asked_for_current_question = False
                        st.session_state.last_question_for_follow_up = ""
                        st.session_state.current_question_index += 1
                        if st.session_state.current_question_index < len(cand["technical_questions"]):
                            replace_placeholder(f"Question {st.session_state.current_question_index+1}: {cand['technical_questions'][st.session_state.current_question_index]}")
                        else:
                            replace_placeholder("That completes the technical questions. Saving your results now.")
                            st.session_state.state = "exit"
            else:
                idx = st.session_state.current_question_index
                if idx >= len(cand["technical_questions"]):
                    replace_placeholder("No more technical questions. Finishing session now.")
                    st.session_state.state = "exit"
                else:
                    if any(skip in lower for skip in SKIP_KEYWORDS):
                        cand["technical_answers"].append("")
                        cand["scores"].append({"question": cand["technical_questions"][idx], "score":None, "justification":"Skipped", "follow_up":False})
                        st.session_state.current_question_index += 1
                        if st.session_state.current_question_index < len(cand["technical_questions"]):
                            replace_placeholder(f"Question {st.session_state.current_question_index+1}: {cand['technical_questions'][st.session_state.current_question_index]}")
                        else:
                            replace_placeholder("That completes the technical questions. Saving your results now.")
                            st.session_state.state = "exit"
                    else:
                        question_text = cand["technical_questions"][idx]
                        rel = llm_check_relevance(llm_main, question_text, text)
                        if not rel["relevant"]:
                            replace_placeholder("That answer seems irrelevant. " + rel["explanation"] + " Please answer again or type 'skip' to move on.")
                        else:
                            cand["technical_answers"].append(text.strip())
                            eval_res = evaluate_answer_with_evaluator(llm_eval, llm_main, question_text, text, cand.get("desired_position","General"), followup_context=False)
                            cand["scores"].append({"question":question_text,"score":eval_res.get("score"),"justification":eval_res.get("justification"),"follow_up":bool(eval_res.get("follow_up")),"synthetic_penalty":eval_res.get("synthetic_penalty",0)})
                            follow_up = eval_res.get("follow_up","").strip()
                            if follow_up and not st.session_state.follow_up_asked_for_current_question:
                                
                                st.session_state.messages.append({"role":"assistant","content":f"Follow-up: {follow_up} (you may answer or type 'skip')","ts":time.time()})
                                st.session_state.expecting_follow_up = True
                                st.session_state.follow_up_asked_for_current_question = True
                                st.session_state.last_question_for_follow_up = question_text
                                replace_placeholder("")
                            else:
                                st.session_state.current_question_index += 1
                                if st.session_state.current_question_index < len(cand["technical_questions"]):
                                    replace_placeholder(f"Question {st.session_state.current_question_index+1}: {cand['technical_questions'][st.session_state.current_question_index]}")
                                else:
                                    replace_placeholder("That completes the technical questions. Saving your results now.")
                                    st.session_state.state = "exit"

        # EXIT: save and finalize
        if st.session_state.state == "exit" and not st.session_state.conversation_ended:
            replace_placeholder("Thank you for your time. We will share results later via email. Goodbye!")
            try:
                save_result = save_results_to_db(st.session_state.candidate)
                if save_result.get("ok"):
                    st.session_state.messages.append({"role":"assistant","content":"✅ Your personal details have been saved.","ts":time.time()})
                else:
                    st.session_state.messages.append({"role":"assistant","content":"⚠️ Personal details were NOT saved.","ts":time.time()})
                if save_result.get("qa_id"):
                    st.session_state.messages.append({"role":"assistant","content":"✅ Your QA results have been saved.","ts":time.time()})
                else:
                    st.session_state.messages.append({"role":"assistant","content":"⚠️ QA results were NOT saved.","ts":time.time()})
                st.session_state["_last_save_result"] = save_result
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":f"Error saving to database: {e}","ts":time.time()})
                st.session_state["_last_save_result"] = {"ok":False,"errors":[str(e)],"tracebacks":{"exception":traceback.format_exc()}}
            st.session_state.conversation_ended = True

    # Spinner
    with st.spinner("Processing..."):
        try:
            process_pending(pending_text)
        except Exception as e:
            replace_placeholder(f"Internal processing error: {e}")
            st.session_state["_last_processing_trace"] = traceback.format_exc()

    # Thinking when bot thinks
    cleaned = []
    for m in st.session_state.messages:
        if m.get("role") == "assistant" and (m.get("content","").strip() == "" or (m.get("thinking") and "Thinking" in m.get("content",""))):
            continue
        cleaned.append(m)
    st.session_state.messages = cleaned
    st.session_state.pending = None
    st.session_state.placeholder_index = None
    st.experimental_rerun()

# Ending of Conversation
if st.session_state.conversation_ended:
    st.markdown("---")
    with st.expander("Collected Candidate Information (session)"):
        st.json(st.session_state.candidate)
    personal_col, qa_col, mongo_status = get_mongo_collections()
    if mongo_status and not mongo_status.get("ok"):
        st.warning("DB connection issue: " + mongo_status.get("msg",""))
        if st.checkbox("Show DB error trace (dev)"):
            st.code(st.session_state.get("_mongo_last_error_trace","No trace available"))
    if "_last_save_result" in st.session_state:
        if st.checkbox("Show last DB save result (dev)"):
            st.json(st.session_state["_last_save_result"])
    if st.button("Start New Conversation"):
        keep = ["_mongo_client","_mongo_init_done","_personal_col","_qa_col","_mongo_status","_mongo_last_error_trace"]
        keys = list(st.session_state.keys())
        for k in keys:
            if k not in keep:
                del st.session_state[k]
        st.experimental_rerun()
