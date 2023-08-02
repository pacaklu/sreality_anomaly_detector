FROM python:3.11-slim

# Set working directory
WORKDIR /sreality_anomaly_detector
ENV PYTHONPATH=/sreality_anomaly_detector

COPY ./requirements/requirements.txt /sreality_anomaly_detector

# install packages
RUN pip install -r requirements.txt

# Copy files
COPY ./scripts /sreality_anomaly_detector/scripts
COPY ./sreality_anomaly_detector /sreality_anomaly_detector/sreality_anomaly_detector

CMD [ "python", "-u", "./scripts/scraper.py"]