import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from pathlib import Path

FILE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageDataset(Dataset):
    def __init__(self, root_dir, vector_db=None):
        self.root_dir = Path(root_dir)
        all_paths = [
            p for p in self.root_dir.rglob("*") if p.suffix.lower() in FILE_EXTENSIONS
        ]
        if vector_db is not None:
            existing = vector_db.exists_batch([str(p) for p in all_paths])
            self.image_paths = [p for p in all_paths if str(p) not in existing]
        else:
            self.image_paths = all_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return image, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, str(img_path)


class ImgEmbeddingEngine:
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    @property
    def vector_dim(self):
        return self.model.config.vision_config.hidden_size

    def extract_image_embeddings(self, dir_path, vector_db, progress_bar, batch_size=8, num_workers=2) -> dict:
        dataset = ImageDataset(dir_path, vector_db=vector_db)

        if len(dataset) == 0:
            return 0

        def collate_fn(batch):
            batch = [b for b in batch if b[0] is not None]
            images, paths = zip(*batch)
            input_images = self.processor(images=list(images), return_tensors="pt", padding=True)
            return input_images, paths

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )

        total_images = len(dataset)
        progress = 0

        with torch.no_grad():
            for inputs, paths in dataloader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.get_image_features(**inputs)
                features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                embeddings = torch.nn.functional.normalize(features, p=2, dim=1)
                vector_db.store_batch(paths, embeddings.cpu().numpy())
                progress += len(inputs["pixel_values"])
                progress_bar.progress(progress / total_images, text=f"Extracted {progress}/{total_images} embeddings")

        return len(dataset)

    def extract_text_embedding(self, prompt: str) -> np.ndarray:
        inputs = self.processor(text=[prompt], return_tensors="pt", padding="max_length", max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            embedding = torch.nn.functional.normalize(features, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy()
