# Cancer Prediction Application
This is an application to predict a cancer patient's diagnosis as "M" (Malignant - Benign) or "B" (Benign - Malignant) using the visual characteristics of the cancer and the average values of these characteristics.

## Tools Used
- Python3.11.0
- scikit-learn
- Flask
- Docker
- DVC
- MLFlow

## Steps to run locally
The application can be run locally using docker, please use the below commands
1. Build docker image:

   docker build -t cancer-predict-app .
2. Run the app:

   docker run -p 8080:8080 cancer-predict-app

Note: Within the container, the server runs on port 8080. If you want to map the port to a different one in your host machine, you need to update the above command accordingly. 

E.g. if you want to run it on port 5000 in the host machine, please use docker run -p 5000:8080 cancer-predict-app
