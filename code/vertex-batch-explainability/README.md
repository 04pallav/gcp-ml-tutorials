# Vertex AI batch explainability (HELOC)

Companion code for the [tutorial](https://github.com/04pallav/gcp-ml-tutorials/blob/main/tutorials/vertex-ai-batch-explainability.md).

```bash
pip install -r requirements.txt
python train_model.py
python prepare_bq_input.py --project-id your-project-id
python run_batch_explain.py --project-id your-project-id --bucket-uri gs://your-bucket/vertex-batch-explain
```
