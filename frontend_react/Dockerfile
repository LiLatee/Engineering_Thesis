## base image
#FROM node:12.2.0-alpine
#
## set working directory
#WORKDIR /usr/app
#
## add `/app/node_modules/.bin` to $PATH
#ENV PATH /app/node_modules/.bin:$PATH
#
## install and cache app dependencies
##COPY package.json /app/package.json
#COPY . /usr/app/
#COPY package.json ./usr/app/
#RUN npm install --silent
#RUN npm install react-scripts@3.0.1 -g --silent
#
## start app
#CMD ["npm", "start"]

#FROM node:8
FROM node:12.2.0-alpine

WORKDIR /app/

# Install dependencies
# COPY package.json yarn.lock /app/

# RUN npm install

# Add rest of the client code
COPY . /app/

RUN npm install --silent
RUN npm install react-scripts@3.0.1 -g --silent

EXPOSE 3000

 CMD npm start