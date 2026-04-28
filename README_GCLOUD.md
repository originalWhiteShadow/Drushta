Google Cloud Run deploy instructions
=================================

Prerequisites
- Install and authenticate the Google Cloud SDK: https://cloud.google.com/sdk/docs/install
- Enable required APIs for your project:

```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

Build & push the container (replace PROJECT_ID and REGION):

```bash
# Set variables
PROJECT_ID=your-gcp-project-id
REGION=us-central1

# Authenticate and set project
gcloud auth login
gcloud config set project $PROJECT_ID

# Build and push image to Artifact/Container Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/drushta .
```

Deploy to Cloud Run:

```bash
gcloud run deploy drushta \
  --image gcr.io/$PROJECT_ID/drushta \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8501
```

Notes
- The repository includes a `Dockerfile` that uses Python 3.11. If you require TensorFlow in Cloud Run, add `tensorflow` and `tensorflow-model-optimization` to `requirements.txt` and ensure the image size/CPU limits are sufficient (TF increases image size significantly).
- If you prefer automated builds on push, enable Cloud Build triggers in the Cloud Console that build the `gcr.io/$PROJECT_ID/drushta` image.

Troubleshooting
- If builds fail due to missing wheels for a package, consider switching to a different base image (e.g., `python:3.11-bullseye`) or removing that dependency for the cloud deployment.
