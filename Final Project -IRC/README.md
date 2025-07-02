#  Docker Guide: Build & Share Your Streamlit App

This guide will help you:

1. Build your own Docker environment for a Streamlit app
2. Run your app locally
3. Export and share the Docker image with others

---

## Project Structure

Ensure you have the following files ready:

```
project-folder/
├── main.py              # Your Streamlit app
├── requirements.txt     # Python dependencies
└── Dockerfile           # Docker build instructions
```

---

## Generate `requirements.txt`

After installing the necessary packages in your Python environment, run:

```bash
pip freeze > requirements.txt
```

> Tip: Clean the list to include only essential packages.

---

## Build the Docker Image

1. Navigate to your project directory:

```bash
cd path/to/project-folder
```

2. Build the image using Docker:

```bash
docker build -t your_image_name .
```

> Example:
```bash
docker build -t my-streamlit-app .
```

---

## Run the Docker Container

### Default (Streamlit on port 8501):

```bash
docker run -p 8501:8501 my-streamlit-app
```

Then open in your browser:
```
http://localhost:8501
```

### Custom External Port (e.g. 8888):

```bash
docker run -p 8888:8501 my-streamlit-app
```

Then access it via:
```
http://localhost:8888
```

> Notes:
> - `-p external:internal` maps host port to container port.
> - Streamlit defaults to port `8501` inside the container.

---

## 📤 Export the Docker Image (Share with Others)

To export the image as a `.tar` file:

```bash
docker save -o my-streamlit-app.tar my-streamlit-app
```

Share it via Google Drive, USB, SCP, etc.

---

## Load the Docker Image (For Recipients)

1. Load the Docker image from the `.tar` file:

```bash
docker load -i my-streamlit-app.tar
```

2. Run the container:

```bash
docker run -p 8501:8501 my-streamlit-app
```

---

## Dockerfile Example

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
```

---

## Summary Workflow

| Step                     | Command |
|--------------------------|---------|
| Create requirements.txt | `pip freeze > requirements.txt` |
| Build image             | `docker build -t my-streamlit-app .` |
| Run container           | `docker run -p 8501:8501 my-streamlit-app` |
| Export image            | `docker save -o my-streamlit-app.tar my-streamlit-app` |
| Import image (others)   | `docker load -i my-streamlit-app.tar` |
| Run container (others)  | `docker run -p 8501:8501 my-streamlit-app` |

---

Happy Dockering! 🐳

