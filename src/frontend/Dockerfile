# Use a lightweight Node.js image as the base
FROM node:lts-slim

# Create and set the working directory
WORKDIR /app

# Install http-server globally
RUN npm install -g http-server

# Copy the HTML file into the container
COPY index.html /app

# Expose port 8000
EXPOSE 8000

# Start the HTTP server to serve the HTML page
CMD ["http-server", "-p", "8000"]
