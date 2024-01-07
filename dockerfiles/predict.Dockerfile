FROM python:3.8-slim

# Set working directory
WORKDIR /sreality_anomaly_detector
ENV PYTHONPATH=/sreality_anomaly_detector

COPY ./requirements/requirements.txt /sreality_anomaly_detector

# For proper loading of model (libgomp1 error)
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# install packages
RUN pip install -r requirements.txt

# Copy files
COPY ./sreality_anomaly_detector /sreality_anomaly_detector/sreality_anomaly_detector

CMD [ "python", "-u", "./sreality_anomaly_detector/predict_all_ids.py"]