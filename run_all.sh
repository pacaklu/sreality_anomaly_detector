cd /home/ec2-user/sreality_anomaly_detector
git pull

# Delete all containers
docker rm -vf $(docker ps -aq)
# Delete all images
docker rmi -f $(docker images -aq)

docker build -f dockerfiles/scrape.Dockerfile  . -t scraper
docker run -v /home/ec2-user/data/:/data/ scraper

docker build -f dockerfiles/training.Dockerfile . -t trainer_image
docker run -v /home/ec2-user/data/:/data -v /home/ec2-user/models/:/models trainer_image

docker build -f dockerfiles/api.Dockerfile . -t api
docker run -v /home/ec2-user/models/:/models -d -p 8000:8000 api
# Can be tested with following command
#curl -X POST http://localhost:8000/predict?input_data=4065768524

docker build -f dockerfiles/predict.Dockerfile . -t prediction_image
docker run -v /home/ec2-user/data/:/data/ --network host prediction_image

# STOP ALL
docker stop $(docker ps -a -q)




