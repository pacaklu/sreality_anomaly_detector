FROM python:3.8

# Set working directory
WORKDIR /sreality_anomaly_detector
ENV PYTHONPATH=/sreality_anomaly_detector

# Copy files
COPY ./requirements/requirements.txt /sreality_anomaly_detector

# install packages
RUN pip install -r requirements.txt

COPY ./api /sreality_anomaly_detector/api
#COPY ./model /sreality_anomaly_detector/model
COPY ./sreality_anomaly_detector /sreality_anomaly_detector/sreality_anomaly_detector


EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]