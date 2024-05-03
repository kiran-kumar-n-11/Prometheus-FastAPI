# Digit Predictor

This project is a FastAPI application for predicting digits from images using a Keras model. It also includes Prometheus for monitoring API usage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kiran-kumar-n-11/Prometheus-FastAPI.git
   ```

2. Navigate to the project directory:
   ```bash
   cd your_repository
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI application and Prometheus using Docker Compose:
   ```bash
   docker-compose up
   ```

2. Access the FastAPI application at `http://localhost:8000` and the Prometheus dashboard at `http://localhost:9090`.

3. Upload an image of a digit to the FastAPI application to get the predicted digit.

## Project Structure

- `DigitPredictor.h5`: Keras model for digit prediction.
- `app.py`: FastAPI code for predicting digits using images.
- `requirements.txt`: List of Python dependencies.
- `Dockerfile`: Docker configuration file.
- `docker-compose.yml`: Docker Compose configuration file.
- `prometheus/prometheus.yml`: Prometheus configuration file.

