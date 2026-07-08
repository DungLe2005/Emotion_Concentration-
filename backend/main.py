from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pipeline import EmotionPipeline

app = FastAPI(title="Emotion & Concentration Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_instance = None

@app.on_event("startup")
async def startup_event():
    global pipeline_instance
    # Load model nặng 1 lần duy nhất lúc khởi động server
    pipeline_instance = EmotionPipeline()

@app.get("/")
def read_root():
    return {"message": "Welcome to Emotion & Concentration Analysis API"}

@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    global pipeline_instance
    try:
        contents = await file.read()
        
        # Đưa ảnh vào chạy qua Pipeline thật
        result = pipeline_instance.run(contents)
        
        return result
    except Exception as e:
        return {"error": str(e)}
