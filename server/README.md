# ASE2025 Evaluation Server

This repository contains a server application for evaluating submissions for the ASE2025 competition. The server accepts submission files in jsonlines format and returns evaluation results as JSON.

## Features

- RESTful API for submission evaluation
- Docker containerization for easy deployment
- Support for different evaluation stages/phases
- Automatic error handling and reporting

## Installation and Setup

### Prerequisites

- Docker and Docker Compose

### Building and Running the Server

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ase2025-server.git
   cd ase2025-server
   ```

2. Build and start the server using Docker Compose:
   ```bash
   docker-compose up -d
   ```

   This will build the Docker image and start the server in detached mode.

3. The server will be available at http://localhost:8000

### Development Setup

If you want to run the server locally for development:

1. Install Poetry (dependency management):
   ```bash
   pip install poetry
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run the server:
   ```bash
   poetry run uvicorn server.app:app --reload
   ```

## API Documentation

### Health Check

```
GET /
```

Returns a simple message indicating that the server is running.

**Response:**
```json
{
  "message": "ASE2025 Evaluation Server is running"
}
```

### Evaluate Submission

```
POST /evaluate
```

Evaluates a submission file against reference data for a specific stage.

**Request:**
- Content-Type: multipart/form-data
- Parameters:
  - `submission_file`: A jsonlines file containing the user's submission
  - `stage`: The stage/phase of the evaluation

**Response:**
```json
{
  "result": [
    {"model1_split": {"Model 1 ChrF Score": 0.75}},
    {"model2_split": {"Model 2 ChrF Score": 0.75}},
    {"model3_split": {"Model 3 ChrF Score": 0.75}},
    {"average_split": {"Average ChrF Score": 0.75}}
  ]
}
```

**Error Response:**
```json
{
  "error": "Evaluation failed: [error message]"
}
```

## Example Usage

Using curl to submit a file for evaluation:

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "submission_file=@path/to/your/submission.jsonl" \
  -F "stage=example"
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/evaluate"
files = {"submission_file": open("path/to/your/submission.jsonl", "rb")}
data = {"stage": "example"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
