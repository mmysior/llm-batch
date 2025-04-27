import logging
from uuid import uuid4

import click
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from batch_cli.models.schemas import OpenAIBatch
from batch_cli.pipelines.inference import process_request
from batch_cli.pipelines.post import parse_batch_jsonl
from batch_cli.utils.general import (
    append_to_jsonl,
    convert_to_df,
    load_jsonl,
    load_jsonl_generator,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------


@click.group()
def cli():
    """
    Batch CLI: A command-line tool for running and managing batch inference jobs
    """
    pass


# ------------------------------------------------------------
# CLI Commands
# ------------------------------------------------------------


@click.command(name="run-anthropic")
@click.argument("file_path", type=click.Path(exists=True))
def run_anthropic(file_path: str) -> None:
    """
    Run a batch of Anthropic requests from a JSONL file.
    """
    anthropic_client = Anthropic()
    requests = [Request(**item) for item in load_jsonl(file_path)]
    message_batch = anthropic_client.messages.batches.create(requests=requests)

    logger.info("Number of requests in batch: %d", len(requests))
    logger.info("Batch ID: %s", message_batch.id)


@click.command(name="run")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--interval", type=int, default=100)
def run(file_path: str, interval: int) -> None:
    """
    Run a batch of OpenAI requests from a JSONL file with Ollama and save the responses to an output file.
    Use --interval to control how often results are written.
    """
    batch_id = str(uuid4().hex)
    output_path = f"batch_{batch_id}_output.jsonl"
    responses = []
    count = 0

    logger.info("Starting batch %s", batch_id)
    for item in load_jsonl_generator(file_path):
        request = OpenAIBatch(**item)
        response = process_request(request, batch_id)
        responses.append(response)
        count += 1

        # Save in batches based on the interval
        if count % interval == 0:
            append_to_jsonl(responses, output_path)
            responses = []  # Clear memory after saving
            logger.info("Saved %d responses to %s", count, output_path)

    # Save any remaining responses
    if responses:
        append_to_jsonl(responses, output_path)

    logger.info("Results saved to %s", output_path)


@click.command(name="parse")
@click.argument("file_path", type=click.Path(exists=True))
def parse(file_path: str) -> str:
    models = parse_batch_jsonl(file_path)
    df = convert_to_df(models)
    output_path = file_path.rsplit(".", 1)[0] + ".csv"
    df.to_csv(output_path, index=False, encoding="utf-8", sep=";")
    return f"File saved to {output_path}"


# ------------------------------------------------------------
# Add commands to the CLI
# ------------------------------------------------------------
cli.add_command(run_anthropic)
cli.add_command(run)
cli.add_command(parse)
