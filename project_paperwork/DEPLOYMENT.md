# Healthcare QA Chatbot - Deployment Guide

## Quick Start Options

| Method | Difficulty | Best For |
|--------|------------|----------|
| **Docker Compose** | Easy | Local/VPS deployment |
| **Railway/Render** | Easy | Free cloud hosting |
| **AWS/GCP** | Medium | Production scale |
| **Hugging Face Spaces** | Easy | Demo/ML showcase |

---

## Option 1: Docker Compose (Recommended)

**Fastest way to deploy on any server with Docker.**

```bash
cd /home/kbs/final_project

# Build and run (first time - takes ~10 min)
docker-compose -f docker/docker-compose.yml up -d --build

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

**Access:**
- Frontend: http://your-server-ip:7860
- API: http://your-server-ip:8000

**On a VPS (DigitalOcean, Linode, AWS EC2):**
```bash
# SSH into your server
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone your project
git clone <your-repo-url> healthcare-qa
cd healthcare-qa

# Run
docker-compose -f docker/docker-compose.yml up -d --build
```

---

## Option 2: Railway (Free Tier Available)

**Deploy directly from GitHub:**

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select your repo
5. Railway auto-detects Dockerfile
6. Add environment variables if needed
7. Deploy!

**railway.json** (create in project root):
```json
{
  "build": {
    "dockerfilePath": "docker/Dockerfile"
  },
  "deploy": {
    "startCommand": "python api/main.py",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

---

## Option 3: Render (Free Tier Available)

1. Push to GitHub
2. Go to [render.com](https://render.com)
3. New → Web Service
4. Connect your repo
5. Settings:
   - Runtime: Docker
   - Dockerfile Path: `docker/Dockerfile`
   - Port: 8000

---

## Option 4: Hugging Face Spaces

**Best for ML demos:**

1. Create account at huggingface.co
2. New Space → Gradio or Streamlit
3. Push your frontend code
4. Add `requirements.txt`

---

## Pre-Deployment Checklist

```bash
# 1. Make sure knowledge base is built
python scripts/build_knowledge_base.py

# 2. Test locally
./venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Verify Docker build works
docker build -f docker/Dockerfile -t healthcare-qa .

# 4. Test Docker container
docker run -p 8000:8000 healthcare-qa
```

---

## Environment Variables

Set these in your deployment platform:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_GPU` | Enable GPU | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HUGGINGFACE_TOKEN` | HF API token | (optional) |

---

## Current Local Deployment

Your app is currently running at:
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000

To keep running after terminal closes:
```bash
# Use nohup or screen
nohup ./venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
nohup streamlit run frontend/streamlit_app.py --server.port 8501 &
```
