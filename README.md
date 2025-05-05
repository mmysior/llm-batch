# 📦 LLM Batch

A command-line tool for running and managing batch inference jobs with LLM providers (OpenAI and Anthropic).

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
uv pip install llm-batch
```

### Using pip

```bash
pip install llm-batch
```

## Usage

The `llm-batch` tool offers several commands to create, run, and parse batches of LLM inference requests.

### Available Commands

- `create`: Create a batch of requests from a CSV or JSON file
- `run`: Run a batch of requests through Ollama (requires Ollama installation with the model specified in your config)
- `run-anthropic`: Run a batch of Anthropic requests directly. Requires an `ANTHROPIC_API_KEY` environmental variable.
- `parse`: Parse and convert batch results to CSV

### Creating a Batch

```bash
llm-batch create INPUT_PATH CONFIG_FILE OUTPUT_PATH
```

- `INPUT_PATH`: Path to a CSV or JSON file containing questions
- `CONFIG_FILE`: Path to the YAML config file (see configuration section)
- `OUTPUT_PATH`: Path to save the output JSONL file

The input CSV file should have at least two columns:
- `question_id`: A unique identifier for the question
- `question`: The text of the question
- `image_path` (optional): Path to an image file for multimodal models

### Running a Batch

```bash
llm-batch run FILE_PATH [--interval INTEGER] [--output-dir DIRECTORY] [--verbose]
```

- `FILE_PATH`: Path to the JSONL file containing the batch
- `--interval`: Number of responses to save at once (default: 100)
- `--output-dir`: Directory to save the output (default: current directory)
- `--verbose`: Enable verbose logging

This command processes the batch requests through Ollama and saves the responses to a JSONL file.

**Important**: This command requires:
1. [Ollama](https://ollama.com/) to be installed on your system
2. The model specified in your config to be downloaded in Ollama `ollama pull <model_name>`

### Running Anthropic Batches Directly

```bash
llm-batch run-anthropic FILE_PATH
```

- `FILE_PATH`: Path to the JSONL file containing Anthropic requests

This command uses Anthropic's native batch API.

### Parsing Results

```bash
llm-batch parse INPUT_PATH [OUTPUT_DIR]
```

- `INPUT_PATH`: Path to the JSONL file containing batch responses
- `OUTPUT_DIR`: Directory to save the parsed CSV file (default: current directory)

This command parses the batch responses and converts them to a CSV file.

## Configuration

The tool uses a YAML configuration file to specify the parameters for creating batch requests. Here's an explanation of the configuration options:

```yaml
# Format of the batch requests
format: openai  # Options: "openai" or "anthropic"

# Number of responses to generate per question
# n_answers: 5    # Uncomment to set a different value (default: 1)

# Generation parameters
params:
  model: gemma2:2b         # Model to use for inference
  temperature: 0.7         # Controls randomness
  max_tokens: 8196         # Maximum tokens to generate

  # Optional parameters
  # top_p: 0.9              # Nucleus sampling parameter
  # frequency_penalty: 0.0  # Penalize repeated tokens
  # presence_penalty: 0.0   # Penalize tokens already present

# System message to include in the prompt (optional)
# system_message: |
#   You are a helpful AI assistant.

# JSON Schema for structured output (optional)
# json_schema:
#   name: response_model        # Name of the schema
#   schema:                     # JSON schema definition
#     type: object
#     properties:
#       thinking:
#         type: string
#       answer:
#         type: string
#     required:
#       - thinking
#       - answer
#     additionalProperties: false
#   strict: true               # Enforce strict schema 
```

### Important Configuration Notes

- **format**: Must be either "openai" or "anthropic" based on which provider you're using
- **params**: Contains model parameters like model, temperature, and token limits
- **json_schema**: Optional JSON schema for structured responses (useful for parsing)

## Environment Variables

The tool requires API keys for the LLM providers you're using:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

You can also use a `.env` file in your project directory.

## Example Workflow

1. Prepare a CSV file with questions (`questions.csv`)
2. Create a config file (`config.yaml`)
3. Create a batch file:
   ```bash
   llm-batch create questions.csv config.yaml batches/my_batch.jsonl
   ```
4. Run the batch:
   ```bash
   # Make sure Ollama is installed and the model is downloaded
   ollama pull <model_name>
   
   # Run the batch
   llm-batch run batches/my_batch.jsonl --output-dir results
   ```
5. Parse the results:
   ```bash
   llm-batch parse results/batch_*.jsonl results
   ```

## License

See the [LICENSE](LICENSE) file for details.
