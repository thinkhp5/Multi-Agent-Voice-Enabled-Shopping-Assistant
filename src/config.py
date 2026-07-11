"""
Configuration: Logger, API clients, and settings.

This module is imported by everything else, so it must have zero
dependencies on other project modules.
"""

import logging
import os
import sys

import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, BedrockEmbeddings

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

# ── Logger ───────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("config")

# ── AWS Configuration ────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
BEDROCK_EMBEDDING_MODEL_ID = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Validate AWS credentials are available
try:
    sts = boto3.client("sts", region_name=AWS_REGION)
    sts.get_caller_identity()
    logger.info("AWS credentials validated (region: %s)", AWS_REGION)
except Exception as e:
    logger.error(
        "AWS credentials not configured. Set up credentials via:\n"
        "  - AWS CLI: aws configure\n"
        "  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
        "  - IAM role (if running on EC2/Lambda)\n"
        "Error: %s", e
    )
    sys.exit(1)

# ── Clients ──────────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
polly_client = boto3.client("polly", region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)

llm = ChatBedrock(
    model_id=BEDROCK_MODEL_ID,
    client=bedrock_runtime,
    model_kwargs={"temperature": 0.3, "max_tokens": 4096},
)
embeddings = BedrockEmbeddings(
    model_id=BEDROCK_EMBEDDING_MODEL_ID,
    client=bedrock_runtime,
)

logger.info(
    "AWS Bedrock clients initialised  (model: %s, embeddings: %s)",
    BEDROCK_MODEL_ID, BEDROCK_EMBEDDING_MODEL_ID,
)
