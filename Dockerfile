#Mention the base image 
#FROM continuumio/anaconda3:4.4.0

FROM python:3.9

#Copy the current folder structure and content to docker folder
COPY . /usr/ML/app

#Expose the port within docker 
EXPOSE 5000

#Set current working directory
WORKDIR /usr/ML/app

#Install the required libraries
RUN pip install -r requirements.txt
RUN mkdir /usr/images

#container start up command
CMD python flask_api.py
