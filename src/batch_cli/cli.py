import csv
import json
import logging
import os
from uuid import uuid4

import click
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from batch_cli.models.schemas import OpenAIBatch, Question
from batch_cli.pipelines.inference import process_request
from batch_cli.pipelines.post import parse_batch_jsonl
from batch_cli.pipelines.pre import create_batch
from batch_cli.utils.general import (
    append_to_jsonl,
    convert_to_df,
    load_config,
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
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=True), default="."
)
def run(file_path: str, interval: int, output_dir: str) -> None:
    """
    Run a batch of OpenAI requests from a JSONL file with Ollama and save the responses to an output file.
    Use --interval to control how often results are written.
    """
    batch_id = str(uuid4().hex)
    output_path = os.path.join(output_dir, f"batch_{batch_id}_output.jsonl")
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
@click.argument("input_path", type=click.Path(exists=True))
@click.argument(
    "output_dir", type=click.Path(file_okay=False, exists=False), default="."
)
def parse(input_path: str, output_dir: str) -> str:
    models = parse_batch_jsonl(input_path)
    df = convert_to_df(models)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    csv_path = os.path.join(output_dir, f"{input_filename}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")
    return f"File saved to {csv_path}"


@click.command(name="create")
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.argument(
    "output-dir", type=click.Path(file_okay=False, exists=False), default="."
)
def create(csv_path: str, config_file: str, output_dir: str) -> str:
    """
    Create a batch .jsonl file from a CSV file using the specified configuration.
    """
    config = load_config(config_file)
    used_kwargs = config.kwargs or {}

    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        questions = [Question(**row) for row in reader]

    batch_content = create_batch(
        questions=questions,
        model=config.model,
        format=config.format,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n_answers=config.n_answers,
        system_message=config.system_message,
        **used_kwargs,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique file name for the batch
    batch_id = str(uuid4().hex)
    output_file = os.path.join(output_dir, f"batch_{batch_id}.jsonl")

    # Save the batch to a JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in batch_content:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    logger.info("Created batch with %d items", len(batch_content))
    logger.info("Batch saved to %s", output_file)

    return f"Batch file saved to {output_file}"


# ------------------------------------------------------------
# Add commands to the CLI
# ------------------------------------------------------------
cli.add_command(run_anthropic)
cli.add_command(run)
cli.add_command(parse)
cli.add_command(create)
