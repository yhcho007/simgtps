FROM node:18-slim
WORKDIR /app
COPY ./gateway /app
RUN npm install
EXPOSE 3000
CMD ["node","server.js"]
