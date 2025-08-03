from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

SPLIT_MARKER = "Answer the following questions and respond with a JSON array of strings containing the answer."

app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/")
async def root():   
    return {"message": "Hello World"}

@app.post("/api")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return JSONResponse(status_code=400, content={"error": "Only .txt files are accepted"})

    content = await file.read()
    text = content.decode("utf-8")

    return {
        "filename": file.filename,
        "length": len(text),
        "content": text.strip()[:200] + "..." if len(text) > 200 else text.strip()
    }
    