import os
import io
import base64
import json
import subprocess
import tempfile
import uuid
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

app = FastAPI()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Setup ---
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompt Loader ---
def load_prompt(name):
    with open(os.path.join("prompts", name), "r", encoding="utf-8") as f:
        return f.read()

task_breaker_prompt = load_prompt("task_breaker.txt")
code_writer_prompt = load_prompt("code_writer.txt")

# --- Helpers ---
def compress_base64_image(b64_str, max_size=100_000):
    """Compress base64 image string until under max_size bytes."""
    img_data = base64.b64decode(b64_str.split(",")[1])
    img = Image.open(io.BytesIO(img_data))
    quality = 90
    while True:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        b64_out = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
        if len(b64_out.encode()) < max_size or quality <= 40:
            return b64_out
        quality -= 5

# --- AI Step 1: Task Breakdown ---
def task_breaker(task: str, file_summaries: List[dict]):
    files_context = "\n".join(
        [f"File: {f['filename']} | Path: {f.get('path')} | Type: {f['type']} | Size: {f['size']} | Preview: {f['preview']}"
         for f in file_summaries]
    )
    full_prompt = f"""{task_breaker_prompt.strip()}

Question:
{task.strip()}

Available Files:
{files_context}

Steps:"""
    resp = gemini_model.generate_content(full_prompt)
    return resp.text.strip()

def code_writer(steps: str, files: List[dict]):
    files_context = "\n".join(
        [f"File: {f['filename']} | AbsolutePath: {f['path']} | Type: {f['type']} | Preview: {f['preview']}"
         for f in files if f.get("path")]
    )
    prompt = f"""{code_writer_prompt.strip()}

Steps:
{steps}

Files available (with absolute paths only):
{files_context}

Rules:
1. Always use the provided AbsolutePath when reading files (not just filenames).
2. Verify files exist before reading.
3. Print column names + sample rows before processing.
4. Optimize plots to be <100 KB.
5. Final output must be valid JSON and printed with json.dumps().
"""
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()


# --- Execution ---
def extract_json_from_stdout(stdout: str) -> str:
    """Extract last valid JSON object from stdout."""
    try:
        return json.loads(stdout)
    except:
        import re
        matches = re.findall(r"\{.*\}", stdout, re.S)
        if matches:
            return json.loads(matches[-1])
        raise ValueError("No valid JSON found in output")

def execute_code_with_retries(code: str, steps: str, files: list, retries: int = 3):
    for attempt in range(retries):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return extract_json_from_stdout(result.stdout.strip())
            else:
                raise RuntimeError(result.stderr.strip())

        except Exception as e:
            if attempt < retries - 1:
                files_context = "\n".join(
                    [f"File: {f['filename']} | AbsolutePath: {f.get('path')} | Type: {f['type']} | Preview: {f['preview']}"
                    for f in files if f.get("path")]
                )

                fix_prompt = f"""
The following Python code failed.

--- Steps ---
{steps}

--- Code ---
{code}

--- Error ---
{str(e)}

--- Files available (with absolute paths only) ---
{files_context}

Please fix the code. Ensure it uses the correct absolute file paths and still outputs JSON with json.dumps().
"""
                code = code_writer(fix_prompt, files)
            else:
                raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


from fastapi import Request, UploadFile, File
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os, uuid, base64, json, shutil

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/")
async def analyze_file(request: Request):
    try:
        form = await request.form()
        saved_files = []
        questions_file = None
        question_text = None

        for field_name, file_obj in form.items():
            if not hasattr(file_obj, "filename"):
                continue

            # Handle questions.txt specially
            if field_name.strip().lower() == "questions.txt":
                question_text = (await file_obj.read()).decode("utf-8", errors="ignore").strip()
                if not question_text:
                    return JSONResponse(status_code=400, content={"error": "questions.txt is empty"})

                questions_file = {
                    "filename": file_obj.filename,
                    "size": len(question_text),
                    "type": file_obj.content_type or "text/plain",
                    "preview": question_text[:500],
                }
                continue  # don’t save yet, just parse

            # Save other files
            content = await file_obj.read()
            safe_name = f"{uuid.uuid4().hex}_{file_obj.filename}"
            save_path = os.path.join(UPLOAD_DIR, safe_name)
            with open(save_path, "wb") as buffer:
                buffer.write(content)

            if (file_obj.content_type and file_obj.content_type.startswith("text/")) or file_obj.filename.endswith((".csv", ".txt", ".json")):
                preview = content.decode("utf-8", errors="ignore")[:500]
            else:
                preview = base64.b64encode(content).decode()[:500]

            saved_files.append({
                "filename": file_obj.filename,
                "path": save_path,
                "size": len(content),
                "type": file_obj.content_type or "application/octet-stream",
                "preview": preview,
            })

        if not questions_file:
            return JSONResponse(status_code=400, content={"error": "Missing required questions.txt"})

        # Step 2: Task Breakdown
        steps = task_breaker(question_text, saved_files)

        # Step 3: Code Writing
        code = code_writer(steps, saved_files)

        parsed_output = execute_code_with_retries(code, steps, saved_files)

        if not isinstance(parsed_output, (dict, list)):
            try:
                parsed_output = json.loads(parsed_output)
            except Exception:
                parsed_output = {"error": "Invalid JSON output", "raw": str(parsed_output)}

        return parsed_output

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# import os
# import shutil

# app = FastAPI()
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/api/")
# async def upload_files(request: Request):
#     # Parse all form-data fields
#     form = await request.form()
#     saved_files = []

#     for field_name, file_obj in form.items():
#         # Only process file uploads (exclude non-file fields if any)
#         if hasattr(file_obj, "filename"):
#             save_path = os.path.join(UPLOAD_DIR, file_obj.filename)
#             with open(save_path, "wb") as buffer:
#                 shutil.copyfileobj(file_obj.file, buffer)
#             saved_files.append(file_obj.filename)

#     return JSONResponse({"status": "success", "saved_files": saved_files})




