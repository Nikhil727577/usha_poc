import pandas as pd
import base64
import clip
import torch
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

device="cuda" if torch.cuda.is_available() else "cpu"
model,preprocess=clip.load('ViT-B/32',device=device)

def decode_base_64(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string)))


def prepare_fan_embeddings(excel_path="usha_real_fans_dataset.xlsx"):
    df = pd.read_excel(excel_path)
    images = [preprocess(decode_base_64(b64)).unsqueeze(0) for b64 in df["Image (base64)"]]
    images = torch.cat(images).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(images)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    df["embedding"] = embeddings.cpu().tolist()
    return df

df_with_embeddings = prepare_fan_embeddings()
embeddings_tensor = torch.tensor(df_with_embeddings["embedding"].tolist()).to(device)

@app.post("/identify_fan")
async def identify_fan(image: UploadFile = File(...)):
    img = Image.open(BytesIO(await image.read())).convert("RGB")
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(img_preprocessed)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarities = (embeddings_tensor @ query_embedding.T).squeeze()
        best_idx = torch.argmax(similarities).item()
        best_match = df_with_embeddings.iloc[best_idx]

    return {
        "matched_fan": best_match["Fan Name"],
        "product_id": best_match["Product ID"],
        "fan_type": best_match["Fan Type"],
        "Manufacturing_Date":best_match["Manufacturing Date"],
        "warranty period":best_match["Warranty Period"],
        "similarity_score": f"{round(similarities[best_idx].item() * 100, 2)}%"
    }

if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)