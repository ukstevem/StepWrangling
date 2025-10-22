# app.py
import mimetypes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make sure .glb serves with a sensible type
mimetypes.add_type("model/gltf-binary", ".glb")

app = FastAPI()

# If you’ll open the viewer from another origin/port, keep CORS permissive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down later if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder layout:
# viewer/
#   index.html      (the file I put on canvas)
#   data/
#     manifest.json
#     defs/...
#     instances/...
#
# Mount viewer at root so http://localhost:8000/ serves index.html
app.mount("/", StaticFiles(directory="viewer", html=True), name="viewer")

# Optional: explicit /data mount (not strictly needed if data/ is inside viewer/)
# app.mount("/data", StaticFiles(directory="viewer/data"), name="data")
