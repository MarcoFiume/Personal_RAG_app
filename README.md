# RAG Image Search Demo

A Retrieval-Augmented Generation (RAG) demo application that allows you to build a local image library and search through it using natural language text prompts. This project uses state-of-the-art vision-language models to create a semantic search engine for your local images.

## 🚀 Features

- **Semantic Search**: Search for images using natural language (e.g., "a red car on a mountain road") instead of just filenames or tags.
- **Efficient Image Embedding**: Uses the **SigLIP 2** model from Google via the Hugging Face Transformers library for high-quality image and text representations.
- **Vector Database**: Utilizes **Redis** with the RediSearch module (HNSW algorithm) for fast similarity search based on cosine distance.
- **Batch Processing**: Supports multi-threaded image processing and GPU acceleration for fast library indexing.
- **Interactive UI**: Built with **Streamlit**, featuring a clean dashboard to manage your library, monitor statistics, and view search results with pagination.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Vector Database**: [Redis Stack](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/) (with RediSearch)
- **Programming Language**: Python 3.x

## 📋 Prerequisites

- **Redis Stack**: You need a running Redis instance with the Search module enabled. The easiest way is via Docker:
  ```bash
  docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
  ```
- **Python Dependencies**: Listed in the code (transformers, torch, redis, streamlit, pillow, numpy).

## 🔧 Setup & Usage

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision transformers redis streamlit pillow numpy
   ```

3. **Configure Settings** (Optional):
   Modify `settings.json` to change the model (default: `google/siglip2-base-patch16-naflex`) or UI defaults.

4. **Run the application**:
   ```bash
   streamlit run main.py
   ```

5. **Using the App**:
   - **Manage image library**: Enter a local directory path to scan and index your images.
   - **Search images**: Go to the search tab and enter any descriptive text prompt.

## 📝 Project Structure

- `main.py`: The entry point of the Streamlit application.
- `inference.py`: Handles model loading and embedding extraction for images and text.
- `db.py`: Manages the connection to Redis and vector search operations.
- `ui.py`: Contains the Streamlit UI components and layout.
- `utils.py`: Utility functions for configuration management.
- `settings.json`: Application configuration parameters.

---
*This is a personal project for my resume, demonstrating integration of deep learning models with vector databases and interactive web interfaces.*
