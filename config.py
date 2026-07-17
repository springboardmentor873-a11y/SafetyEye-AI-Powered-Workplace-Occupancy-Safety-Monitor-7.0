import os

# PostgreSQL connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "safety_eye")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Video processing
FRAME_SAMPLE_INTERVAL = int(os.getenv("FRAME_SAMPLE_INTERVAL", "30"))  # process every Nth frame
FRAMES_OUTPUT_DIR = os.getenv("FRAMES_OUTPUT_DIR", "frames")           # where annotated JPEGs are saved

# ONNX model (no PyTorch required at runtime)
MODEL_PATH = os.getenv("MODEL_PATH", "model/weights/best.onnx")

# Violation class names (derived from construction-ppe dataset)
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots", "none"}

SEVERITY_MAP = {
    "none": "high",       # no gear at all
    "no_helmet": "medium",
    "no_goggle": "medium",
    "no_gloves": "medium",
    "no_boots": "medium",
}
