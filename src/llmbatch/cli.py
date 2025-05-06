import csv
import json
import logging
import os
from uuid import uuid4

import click
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from llmbatch.models.schemas import OpenAIBatch, Question
from llmbatch.pipelines.inference import process_request
from llmbatch.pipelines.post import parse_batch_jsonl
from llmbatch.pipelines.pre import create_batch
from llmbatch.utils.general import (
    append_to_jsonl,
    load_config,
    load_jsonl,
    load_jsonl_generator,
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
    load_dotenv(find_dotenv(usecwd=True))


# ------------------------------------------------------------
# CLI Commands
# ------------------------------------------------------------


@click.command(name="run-anthropic")
@click.argument("file_path", type=click.Path(exists=True))
def run_anthropic(file_path: str) -> None:
    """
    Run a batch of Anthropic requests from a JSONL file.
    """
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
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
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
def run(file_path: str, interval: int, output_dir: str, verbose: bool) -> None:
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    batch_id: str = str(uuid4().hex)
    output_path: str = os.path.join(output_dir, f"batch_{batch_id}_output.jsonl")
    responses: list = []
    count: int = 0

    total_items: int = sum(1 for _ in load_jsonl_generator(file_path))

    logger.info("Starting batch %s", batch_id)

    pbar = tqdm(total=total_items, desc="Processing batch", unit="requests")

    for item in load_jsonl_generator(file_path):
        request = OpenAIBatch(**item)
        response = process_request(request, batch_id)
        responses.append(response)
        count += 1
        pbar.update(1)

        if count % interval == 0:
            append_to_jsonl(responses, output_path)
            responses = []
            pbar.set_description(f"Processing batch (saved {count} responses)")

    if responses:
        append_to_jsonl(responses, output_path)

    pbar.close()
    logger.info("Results saved to %s", output_path)


@click.command(name="parse")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument(
    "output_dir", type=click.Path(file_okay=False, exists=False), default="."
)
def parse(input_path: str, output_dir: str) -> str:
    """
    Parse a batch of responses from a JSONL file and save the results as a CSV file.
    """
    models = parse_batch_jsonl(input_path)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    csv_path = os.path.join(output_dir, f"{input_filename}.csv")

    # Handle empty results
    if not models:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")
        return f"File saved to {csv_path} (empty results)"

    # Extract field names from the first model
    fieldnames = list(models[0].model_dump().keys())

    # Write directly to CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for model in models:
            writer.writerow(model.model_dump())

    return f"File saved to {csv_path}"


@click.command(name="create")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=False))
def create(input_path: str, config_file: str, output_path: str) -> str:
    """
    Create a batch of requests from a CSV file with question_id and question columns and save the results as a JSONL file.
    The config file must be in .csv format.
    """
    config = load_config(config_file)
    used_kwargs = config.params.model_dump()
    # Handle response_model unpacking if present in config
    if config.json_schema:
        if config.format == "openai":
            used_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": config.json_schema,
            }
        elif config.format == "anthropic":
            used_kwargs["tools"] = [
                {
                    "name": config.json_schema.get("name", "response_model"),
                    "description": "Respond with a JSON object describing an action with its positive and negative effects.",
                    "input_schema": config.json_schema.get("schema", {}),
                }
            ]
            used_kwargs["tool_choice"] = {
                "type": "tool",
                "name": config.json_schema.get("name", "response_model"),
            }

    questions: list[Question] = []
    if input_path.endswith(".csv"):
        with open(input_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            questions = [Question(**row) for row in reader]
    elif input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            questions = [Question(**item) for item in data]
    else:
        raise ValueError("Input file must be a .csv or .json file.")

    batch_content = create_batch(
        questions=questions,
        format=config.format,
        n_answers=config.n_answers,
        system_message=getattr(config, "system_message", None),
        **used_kwargs,
    )

    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        batch_id = str(uuid4().hex)
        output_file = os.path.join(output_path, f"batch_{batch_id}.jsonl")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_file = output_path

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
