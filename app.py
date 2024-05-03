from fastapi import FastAPI, File, UploadFile, Request
from tensorflow import keras
from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import io
import time
import uvicorn
from prometheus_client import Counter, Gauge, Summary, start_http_server
from prometheus_client import disable_created_metrics

# disable _created metric.
disable_created_metrics()


# Define Prometheus metrics
REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
api_usage_counter = Counter("api_usage", "API usage counter", ['endpoint','client_ip'])
api_runtime_gauge = Gauge("api_runtime", "API runtime gauge", ['endpoint', 'client_ip'])
api_tltime_gauge = Gauge("api_tltime", "API T/L time gauge", ['endpoint', 'client_ip'])


app = FastAPI()

#Function to load keras model for digit prediction
def load_keras_model(path: str) -> Sequential:
    return load_model(path)

#function to make prediction from array of grayscale values
def predict_digit(model, data_point: list) -> str:
    data_array = np.array(data_point,dtype=np.float64)/ 255.0
    prediction = model.predict(data_array.reshape(1,-1))
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

#to resize the image
def format_image(file: UploadFile):
    img_bytes = file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28,28))
    img_array = np.array(img).reshape(784)
    return img_array

@REQUEST_DURATION.time()
@app.post("/predict")
async def predict(request:Request,file: UploadFile = File(...)):

    start_time = time.time()

    img_array = format_image(file)
    model_path = "DigitPredictor.h5"
    model = load_keras_model(model_path)
    digit = predict_digit(model, img_array)

    end_time = time.time()
    runtime = end_time - start_time
    if len(file.file.read()):
        tltime = runtime / len(file.file.read())
    else:
        tltime = runtime
    
    # Increment usage counter for the client IP
    api_usage_counter.labels(endpoint="/predict",client_ip=request.client.host).inc()

    # Set runtime and T/L time gauges
    api_runtime_gauge.labels(endpoint="/predict", client_ip=request.client.host).set(runtime)
    api_tltime_gauge.labels(endpoint="/predict", client_ip=request.client.host).set(tltime)


    return {"digit": digit}


if __name__=="__main__":
    # start the exporter metrics service
    start_http_server(18000)

    # Run from command line: uvicorn ai_app:app --port 7000 --host 0.0.0.0
    # or invoke the code below.
    uvicorn.run(app, host='0.0.0.0', port=8000)
