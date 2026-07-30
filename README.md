# ☁️ GCP ML Tutorials

Public tutorials and runnable code for Google Cloud Machine Learning.

## Vertex AI Batch Explainability

**Article:** [tutorials/vertex-ai-batch-explainability.md](tutorials/vertex-ai-batch-explainability.md)

**Code:** [code/vertex-batch-explainability/](code/vertex-batch-explainability/)

```bash
git clone https://github.com/04pallav/gcp-ml-tutorials.git
cd gcp-ml-tutorials/code/vertex-batch-explainability
pip install -r requirements.txt
python batch_explain.py --project-id your-project-id --bucket-uri gs://your-bucket/vertex-batch-explain --instances gs://your-bucket/instances.json
```
