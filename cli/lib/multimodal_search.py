from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.cosine_similarity import cosine_similarity
from lib.search_utils import load_movies


class MultiModalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_fpath):
        img = Image.open(image_fpath)
        return self.model.encode(img)

    def search_with_image(self, image_fpath, limit=5):
        image_embedding = self.embed_image(image_fpath=image_fpath)
        similarities = []
        for idx, text_embedding in enumerate(self.text_embeddings):
            similarities.append(
                (idx, cosine_similarity(image_embedding, text_embedding))
            )

        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        sorted_similarities = sorted_similarities[:limit]
        results = []
        for idx, score in sorted_similarities:
            _doc = self.documents[idx]
            results.append(
                {
                    "title": _doc["title"],
                    "description": _doc["description"],
                    "doc_id": idx,
                    "score": score,
                }
            )
        return results


def verify_image_embedding(image_fpath):
    print("Loading CLIP model (this may take a while on first run)...")
    model = SentenceTransformer("clip-ViT-B-32")
    print(f"Encoding image: {image_fpath}")
    img = Image.open(image_fpath)
    embedding = model.encode(img)
    print(f"✓ Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_fpath, limit=5):
    print("Loading movies and CLIP model...")
    movies = load_movies()
    ms = MultiModalSearch(movies)
    print(f"Searching for movies similar to image: {image_fpath}\n")
    result = ms.search_with_image(image_fpath=image_fpath, limit=limit)
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} (similarity: {r['score']:.4f})")
        print(f"    {r['description'][:100]}...")
