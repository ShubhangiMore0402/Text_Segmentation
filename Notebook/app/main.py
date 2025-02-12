import uvicorn
from fastapi import FastAPI, UploadFile, File
import torch
from model import load_model
from utils import preprocess_image
from PIL import Image
import numpy as np
import io

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load model
device = "cpu"
model = load_model(device=device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # ✅ Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # ✅ Preprocess image
    image_tensor = preprocess_image(np.array(image)).to(device)

    # ✅ Model inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # ✅ Convert output to binary mask
    mask = (torch.sigmoid(output) > 0.5).cpu().numpy().squeeze()

    return {"prediction": mask.tolist()}

# ✅ Run server (Only for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
