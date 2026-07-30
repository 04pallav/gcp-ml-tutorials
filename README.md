# ☁️ GCP ML Tutorials

Public tutorials and runnable code for Google Cloud Machine Learning.

## Vertex AI Batch Explainability

**Tutorial + code:** [code/vertex-batch-explainability/](code/vertex-batch-explainability/)

```bash
git clone https://github.com/04pallav/gcp-ml-tutorials.git
cd gcp-ml-tutorials/code/vertex-batch-explainability
pip install -r requirements.txt
gcloud storage cp instances.json gs://your-bucket/instances.json
python batch_explain.py --project-id your-project-id --bucket-uri gs://your-bucket/vertex-batch-explain --instances gs://your-bucket/instances.json
```
