"""mindcare-ai data pipeline.

Stage-based pipeline that downloads mental-health-related datasets from
HuggingFace, normalizes them into a single schema, flags crisis content,
splits deterministically, and produces a data quality report.

Run:
    python -m src.data_pipeline.run_pipeline --stage all
"""

__version__ = "0.1.0"