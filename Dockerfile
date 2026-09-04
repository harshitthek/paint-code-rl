FROM node:20.17.0-bullseye-slim

# Install Chromium dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    chromium \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

WORKDIR /app

# Node dependencies
COPY renderer/package.json renderer/package-lock.json* ./renderer/
RUN cd renderer && npm ci

# Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy source
COPY . .

# Run the API service
EXPOSE 3000
CMD ["node", "renderer/server.js"]
