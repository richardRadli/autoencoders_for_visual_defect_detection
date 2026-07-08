FROM node:20-slim

WORKDIR /app

# Leverages caching strategies for node modules allocations
COPY package*.json ./
RUN npm install

# Copy all files from the frontend sub-tree
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]