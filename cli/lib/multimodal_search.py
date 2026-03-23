from PIL import Image
from sentence_transformers import SentenceTransformer


class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_fpath):
        img = Image.open(image_fpath)
        return self.model.encode(img)


def verify_image_embedding(image_fpath):
    print("Loading CLIP model (this may take a while on first run)...")
    ms = MultiModalSearch()
    print(f"Encoding image: {image_fpath}")
    embedding = ms.embed_image(image_fpath)
    print(f"✓ Embedding shape: {embedding.shape[0]} dimensions")
