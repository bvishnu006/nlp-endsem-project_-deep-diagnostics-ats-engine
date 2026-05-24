import os
import re
import json
import torch
from collections import Counter

# Document Extraction
import fitz  # PyMuPDF
import docx

# NLP & Scoring
import language_tool_python
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Gemini API
from google import genai
from google.genai import types

# Web UI Layer
import gradio as gr

# ============================================================
# INITIALIZE GLOBAL MODELS (Run once on startup)
# ============================================================
print("⏳ Phase 1/3: Loading Zero-Shot Classification Model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("⏳ Phase 2/3: Loading Grammar Evaluation Suite...")
tool = language_tool_python.LanguageTool('en-US')

print("⏳ Phase 3/3: Loading SBERT Semantic Model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("✅ Initialization Complete! Starting Web Server...")

# Global category anchors
JD_TARGET_CATEGORIES = ["Core Skills", "Responsibilities", "Qualifications", "Education", "Ignore"]
RESUME_TARGET_CATEGORIES = ["Education", "Skills", "Projects", "Experience", "Ignore"]

# ============================================================
# FILE TEXT EXTRACTORS (Adapted for Gradio File Paths)
# ============================================================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return f"Unsupported format: {ext}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return text

def is_potential_header(line):
    clean_line = line.strip()
    if not clean_line: return False
    words = clean_line.split()
    if len(words) > 5: return False
    if clean_line.endswith(('.', ',', ';')): return False
    if clean_line.isupper() or clean_line.istitle(): return True
    return False

# ============================================================
# PARSING & SEGMENTATION ENGINES
# ============================================================
def dynamic_segment_jd(text, progress=gr.Progress()):
    sections = {cat: "" for cat in JD_TARGET_CATEGORIES}
    current_section = "Ignore"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue
        if is_potential_header(clean_line):
            result = classifier(
                clean_line,
                JD_TARGET_CATEGORIES,
                hypothesis_template="This text line is explicitly a job description section title for {}."
            )
            if result['scores'][0] > 0.40:
                current_section = result['labels'][0]
                continue
        sections[current_section] += clean_line + "\n"
    return sections

def dynamic_segment_resume(text, progress=gr.Progress()):
    sections = {cat: "" for cat in RESUME_TARGET_CATEGORIES}
    current_section = "Ignore"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue
        if is_potential_header(clean_line):
            result = classifier(
                clean_line,
                RESUME_TARGET_CATEGORIES,
                hypothesis_template="This text line is explicitly a resume section title for {}."
            )
            if result['scores'][0] > 0.40:
                current_section = result['labels'][0]
                continue
        sections[current_section] += clean_line + "\n"
    return sections

# ============================================================
# DIAGNOSTIC METRIC ENGINES
# ============================================================
def get_grammar_score(text):
    if not text.strip(): return 0, "Low: No text provided."
    test_text = text[:2000]
    matches = tool.check(test_text)
    total_words = len(test_text.split())
    if total_words == 0: return 0, "Low: No valid words found."
    score = round(max(0, 100 - ((len(matches) / total_words) * 100)), 2)
    reasoning = "High: Excellent syntax." if score >= 90 else f"Medium: Minor structural errors rules hit: {', '.join(set([m.ruleId for m in matches[:2]]))}." if score >= 70 else "Low: Substantial grammar issues flagged."
    return score, reasoning

def get_formatting_score(text):
    score = 100
    lines = text.split("\n")
    bullets = [line.strip()[0] for line in lines if line.strip().startswith(("-", "•", "*", ""))]
    if not bullets:
        return 85, "Low: No bullet points detected. Structure is heavy to parse visually."
    most_common_ratio = Counter(bullets).most_common(1)[0][1] / len(bullets)
    if most_common_ratio < 0.7:
        return 90, "Medium: Mixed bullet-point styles used."
    return 100, "High: Clean, uniform visual formatting structures."

def calculate_semantic_matrix(resume_text, jd_text, section_name="Section"):
    if not resume_text.strip() or len(resume_text) < 10:
        return 0.0, f"Missing: Entire {section_name} profile data is empty."
    if not jd_text.strip() or len(jd_text) < 10:
        return 0.0, "Missing: Base target data empty."

    jd_sentences = [s.strip() for s in jd_text.replace('\n', '.').split('.') if len(s.strip()) > 15]
    resume_bullets = [s.strip() for s in resume_text.split('\n') if len(s.strip()) > 5]

    if not jd_sentences or not resume_bullets: return 0.0, "Insufficient structural strings to calculate semantic weight."

    jd_embeddings = semantic_model.encode(jd_sentences, convert_to_tensor=True)
    resume_embeddings = semantic_model.encode(resume_bullets, convert_to_tensor=True)

    cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings)
    max_scores, best_match_indices = torch.max(cosine_scores, dim=1)

    curved_score = round(min(100.0, torch.mean(max_scores).item() * 130), 2)

    missing_indices = (max_scores < 0.45).nonzero(as_tuple=True)[0]
    missing_summary = ""
    if len(missing_indices) > 0:
        missing_summary = "<br><b>Critical Missing Vectors:</b>"
        for idx in missing_indices[:2]:
            missing_summary += f"<br>• <i>{jd_sentences[idx.item()][:75]}...</i>"

    reasoning = f"Match Confidence: {curved_score}% {missing_summary}"
    return curved_score, reasoning

def calculate_dynamic_fusion(scores, has_experience):
    W = {"grammar": 0.05, "format": 0.05, "skills": 0.35, "projects": 0.35, "experience": 0.20}
    if not has_experience:
        exp_weight = W["experience"]
        W["experience"] = 0.0
        total_pool = W["skills"] + W["projects"]
        W["skills"] += exp_weight * (W["skills"] / total_pool)
        W["projects"] += exp_weight * (W["projects"] / total_pool)

    final_score = (
        (scores["Grammar"] * W["grammar"]) +
        (scores["Formatting"] * W["format"]) +
        (scores["Skill_Match"] * W["skills"]) +
        (scores["Project_Match"] * W["projects"]) +
        (scores["Experience_Match"] * W["experience"])
    )
    return round(final_score, 2), W

# ============================================================
# GEMINI INTELLIGENT ROUTER
# ============================================================
def call_gemini_api(resume_dict, jd_dict, ats_dict):
    client = genai.Client(api_key="AIzaSyC1hgOcpj8DiqI69_asRMniq5XTcy_6gYQ")
    prompt = f"""
    RESUME: {json.dumps(resume_dict)}
    JD: {json.dumps(jd_dict)}
    ATS DIAGNOSTICS: {json.dumps(ats_dict)}

    You are an expert ATS evaluator. Output ONLY valid JSON matching this schema exactly without markdown formatting wraps:
    {{
      "summary": {{
        "match_level": "Poor | Moderate | Strong | Excellent",
        "overall_comment": "Detailed structural insights text"
      }},
      "strengths": {{
        "items": [
          {{ "category": "Keyword Relevance / Technical Alignment", "evidence": "specific line", "impact": "why it matters" }}
        ]
      }},
      "missing_requirements": {{
        "critical_missing": ["Missing core tool, skill, or scope requirement"]
      }},
      "resume_improvements": {{
        "high_priority": [
          {{ "issue": "Problem description", "recommended_fix": "Clear descriptive blueprint instruction" }}
        ]
      }}
    }}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    return json.loads(response.text)

# ============================================================
# GRADIO WEB APP PIPELINE
# ============================================================
def process_pipeline(resume_file, jd_file, progress=gr.Progress()):
    if not resume_file or not jd_file:
        return "<h3 style='color:red;'>❌ Please upload both Resume and JD files.</h3>"

    progress(0.1, desc="Extracting text from files...")
    resume_raw_text = extract_text(resume_file.name)
    jd_raw_text = extract_text(jd_file.name)

    if not resume_raw_text.strip() or not jd_raw_text.strip():
        return "<h3 style='color:red;'>❌ Failed to extract text. Ensure documents are not scanned images.</h3>"

    progress(0.3, desc="Segmenting documents via NLP Classifier...")
    resume_segmented = dynamic_segment_resume(resume_raw_text, progress)
    jd_segmented = dynamic_segment_jd(jd_raw_text, progress)

    master_jd_text = "\n".join([v for k, v in jd_segmented.items() if k != "Ignore" and v])
    master_resume_text = "\n".join([v for k, v in resume_segmented.items() if k != "Ignore" and v])

    progress(0.6, desc="Calculating diagnostic matrices...")
    grammar_score, grammar_reasoning = get_grammar_score(master_resume_text)
    format_score, format_reasoning = get_formatting_score(master_resume_text)
    overall_baseline, _ = calculate_semantic_matrix(master_resume_text, master_jd_text, "Overall")
    skill_match, skill_reasoning = calculate_semantic_matrix(resume_segmented.get("Skills", ""), master_jd_text, "Skills")
    project_match, project_reasoning = calculate_semantic_matrix(resume_segmented.get("Projects", ""), master_jd_text, "Projects")

    exp_text = resume_segmented.get("Experience", "")
    has_experience = len(exp_text.strip()) > 15
    experience_match, exp_reasoning = calculate_semantic_matrix(exp_text, master_jd_text, "Experience") if has_experience else (0.0, "Zero experience detected.")

    scores = {
        "Grammar": grammar_score, "Formatting": format_score, "Skill_Match": skill_match,
        "Project_Match": project_match, "Experience_Match": experience_match
    }
    final_ats_score, active_weights = calculate_dynamic_fusion(scores, has_experience)

    final_report_dict = {
        "overall_baseline_score": overall_baseline,
        "final_ats_score": final_ats_score,
        "metrics": {
            "grammar": {"score": scores['Grammar'], "reasoning": grammar_reasoning},
            "formatting": {"score": scores['Formatting'], "reasoning": format_reasoning},
            "skill_relevance": {"score": scores['Skill_Match'], "reasoning": skill_reasoning},
            "project_relevance": {"score": scores['Project_Match'], "reasoning": project_reasoning},
            "experience_relevance": {"score": scores['Experience_Match'], "reasoning": exp_reasoning}
        }
    }

    progress(0.8, desc="Streaming telemetry to Gemini AI...")
    try:
        llm_response = call_gemini_api(resume_segmented, jd_segmented, final_report_dict)
    except Exception as e:
        return f"<h3 style='color:red;'>❌ API Error: {str(e)}</h3>"

    progress(1.0, desc="Rendering Dashboard...")

    # Render UI
    score_color = "#22c55e" if final_ats_score >= 75 else "#eab308" if final_ats_score >= 50 else "#ef4444"

    dashboard_html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="display: flex; gap: 20px; align-items: center; background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 20px;">
            <div style="background: {score_color}; color: white; min-width: 110px; height: 110px; border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center; font-weight: bold; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <span style="font-size: 28px;">{final_ats_score}</span>
                <span style="font-size: 11px; text-transform: uppercase; opacity: 0.9;">ATS Score</span>
            </div>
            <div style="flex: 1;">
                <div style="display: inline-block; padding: 4px 12px; background: #e0f2fe; color: #0369a1; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">
                    Match Quality: {llm_response.get('summary', {}).get('match_level', 'N/A')}
                </div>
                <h3 style="margin: 0 0 8px 0; color: #0f172a;">Executive Evaluation Summary</h3>
                <p style="margin: 0; color: #475569; font-size: 14px; line-height: 1.5;">{llm_response.get('summary', {}).get('overall_comment', '')}</p>
            </div>
        </div>

        <h4 style="color: #334155; margin: 0 0 10px 0; border-left: 4px solid #38bdf8; padding-left: 8px;">Algorithmic Scoring Matrix</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 25px;">
            <div style="background: white; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Skills Alignment</span>
                <div style="font-size: 22px; font-weight: bold; color: #0f172a; margin: 4px 0;">{scores['Skill_Match']}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                <p style="font-size: 11px; color: #64748b; margin: 0; line-height: 1.3;">{skill_reasoning.split('<br>')[0]}</p>
            </div>
            <div style="background: white; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Project Relevance</span>
                <div style="font-size: 22px; font-weight: bold; color: #0f172a; margin: 4px 0;">{scores['Project_Match']}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                <p style="font-size: 11px; color: #64748b; margin: 0; line-height: 1.3;">{project_reasoning.split('<br>')[0]}</p>
            </div>
            <div style="background: white; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Experience Match</span>
                <div style="font-size: 22px; font-weight: bold; color: #0f172a; margin: 4px 0;">{scores['Experience_Match'] if has_experience else 'N/A'}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                <p style="font-size: 11px; color: #64748b; margin: 0; line-height: 1.3;">{exp_reasoning.split('<br>')[0]}</p>
            </div>
            <div style="background: white; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Formatting</span>
                <div style="font-size: 22px; font-weight: bold; color: #0f172a; margin: 4px 0;">{scores['Formatting']}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                <p style="font-size: 11px; color: #64748b; margin: 0; line-height: 1.3;">{format_reasoning}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 12px 0; color: #16a34a; font-size: 15px;">✓ Identified Strengths</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.6;">
    """
    for item in llm_response.get('strengths', {}).get('items', []):
        dashboard_html += f"<li><b>{item.get('category', '')}:</b> {item.get('evidence', '')} <br><span style='color:#64748b; font-size:12px;'>Impact: {item.get('impact', '')}</span></li><br>"

    dashboard_html += """
                    </ul>
                </div>
                <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 12px 0; color: #dc2626; font-size: 15px;">✗ Critical Missing Requirements</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 13px; color: #334155; line-height: 1.6;">
    """
    for gap in llm_response.get('missing_requirements', {}).get('critical_missing', []):
        dashboard_html += f"<li style='margin-bottom: 4px;'>{gap}</li>"

    dashboard_html += """
                    </ul>
                </div>
            </div>
            <div style="background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; height: fit-content;">
                <h4 style="margin: 0 0 12px 0; color: #2563eb; font-size: 15px;">🛠 Recommended Action Blueprint</h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
    """
    for action in llm_response.get('resume_improvements', {}).get('high_priority', []):
        dashboard_html += f"""
        <div style="padding: 10px; background: #f8fafc; border-left: 3px solid #2563eb; border-radius: 0 4px 4px 0;">
            <div style="font-size: 13px; font-weight: 600; color: #1e293b;">Issue: {action.get('issue', '')}</div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px; line-height: 1.4;"><b>Fix:</b> {action.get('recommended_fix', '')}</div>
        </div>
        """
    dashboard_html += """
                </div>
            </div>
        </div>
    </div>
    """
    return dashboard_html

# ============================================================
# GRADIO INTERFACE CONSTRUCTION
# ============================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 8px; text-align: center; color: white; margin-bottom: 20px;">
            <h1 style="margin: 0; color: #38bdf8;">Deep Diagnostics ATS Engine</h1>
            <p style="margin: 5px 0 0 0; color: #94a3b8;">Upload your Resume and Job Description to receive a comprehensive evaluation.</p>
        </div>
    """)

    with gr.Row():
        resume_input = gr.File(label="📄 Upload Resume (.pdf, .docx)", file_types=[".pdf", ".docx"])
        jd_input = gr.File(label="📄 Upload Job Description (.pdf, .docx)", file_types=[".pdf", ".docx"])

    analyze_btn = gr.Button("🚀 Execute Profile Diagnosis", variant="primary", size="lg")

    output_dashboard = gr.HTML(label="Results")

    analyze_btn.click(
        fn=process_pipeline,
        inputs=[resume_input, jd_input],
        outputs=[output_dashboard]
    )

# Launch the web app
demo.launch(debug=True, share=True)
