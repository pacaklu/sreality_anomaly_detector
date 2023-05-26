FROM python:3.8

# Set working directory
WORKDIR /sreality_anomaly_detector
ENV PYTHONPATH=/sreality_anomaly_detector

# Copy files
COPY ./scripts /sreality_anomaly_detector/scripts
COPY ./requirements/requirements.txt /sreality_anomaly_detector

# install packages
RUN pip install -r requirements.txt


CMD [ "python", "./scripts/scraper.py"]