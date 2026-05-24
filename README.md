# NLP-Endsem-project_-Deep-Diagnostics-ATS-Engine
This repository hosts an advanced, end-to-end AI-driven Applicant Tracking System (ATS) pipeline. The engine leverages zero-shot classification for dynamic structural segmentation, Sentence Transformers for semantic vector space embedding matching.

## 🛠️ Installation & Getting Started

Follow these steps to configure your environment and spin up the diagnostic web application.

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install the required system libraries, deep learning frameworks, and NLP packages:

```bash
pip install pymupdf python-docx transformers sentence-transformers language-tool-python google-genai gradio

2. Configure Environment Variable
Set your Google Gemini API key before starting the application:

Linux/macOS: export GEMINI_API_KEY="your-api-key-here"

Windows (CMD): set GEMINI_API_KEY=your-api-key-here

3. Run the Application
Launch the local server and web UI by executing:

Bash
python app.py
Open the generated local address (typically http://127.0.0.1:7860) in your web browser.

🏗️ Technical Architecture
The execution pipeline processes data across four distinct phases:

[File Ingestion] ➔ [Dynamic NLP Segmentation] ➔ [Diagnostic Metric Evaluation] ➔ [Gemini AI Fusion Layer]
Phase 1: Ingestion & Document Parsing
Extraction: Uses PyMuPDF (fitz) and python-docx to extract text from PDF and Word files.

Header Detection: Applies basic structural rules (is_potential_header) to catch section boundaries based on casing and line length.

Phase 2: Zero-Shot Tokenized Segmentation
Classification: Routes headers through a BART-Large-MNLI model to predict section intent.

Mapping: Segregates text into structured categories (Skills, Projects, Experience, Education) using a split confidence threshold score (>0.40).

Phase 3: Multi-Dimensional Metrics Calculation
Grammar & Layout: Evaluates spelling/syntax density using LanguageTool and assesses bullet uniformity via regular expressions.

Semantic Vector Alignment: Uses all-MiniLM-L6-v2 to compute Cosine Similarity between resume text fragments and the JD.

Dynamic Weight Fusion: Automatically shifts experience scoring weight to academic projects and technical skills if zero corporate experience is detected.

Phase 4: Gemini LLM Fusion Layer
Deductive Synthesis: Passes the calculated metrics and segmented raw text profiles to gemini-2.5-flash to return structured JSON.

Dashboard Materialization: Transforms the consolidated intelligence scores into an interactive HTML/Tailwind web dashboard.
