
# docker build -f dockerfiles/scrape.Dockerfile  . -t scraper
docker run -v /home/ec2-user/data/:/data/ scraper

# docker build -f dockerfiles/training.Dockerfile . -t trainer_image
docker run -v /home/ec2-user/data/:/data -v /home/ec2-user/models/:/models trainer_image

# docker build -f dockerfiles/api.Dockerfile . -t api
docker run -v /home/ec2-user/models/:/models -d -p 8000:8000 api
# TEST with following command
#curl -X POST http://localhost:8000/predict?input_data=4065768524



# STOP ALL
docker stop $(docker ps -a -q)