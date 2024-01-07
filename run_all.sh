#!/bin/sh

USE_API=False

cd /home/ec2-user/sreality_anomaly_detector
git pull

# Delete all containers
docker rm -vf $(docker ps -aq)
# Delete all images
docker rmi -f $(docker images -aq)

echo "Building scraper image."
docker build -f dockerfiles/scrape.Dockerfile  . -t scraper
echo "Running scraper image."
docker run -v /home/ec2-user/data/:/data/ scraper

echo "Building model training image."
docker build -f dockerfiles/training.Dockerfile . -t trainer_image
echo "Running model trainer image."
docker run -v /home/ec2-user/data/:/data -v /home/ec2-user/models/:/models trainer_image

if [$USE_API = True] ;
then:
  echo "Building model api image."
  docker build -f dockerfiles/api.Dockerfile . -t api
  echo "Running model api image."
  docker run -v /home/ec2-user/models/:/models -d -p 8000:8000 api
fi

# Can be tested with following command
#curl -X POST http://localhost:8000/predict?input_data=4065768524

echo "Building model predictions image."
docker build -f dockerfiles/predict.Dockerfile . -t prediction_image
echo "Running model predictions image."
docker run -v /home/ec2-user/data/:/data/ --network host prediction_image

# STOP ALL
echo "Stopping all containers."
docker stop $(docker ps -a -q)




