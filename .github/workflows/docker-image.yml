# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Copy the shell script into the container
COPY run_during_business_hours.sh /app/run_during_business_hours.sh

# Give the shell script execution permissions
RUN chmod +x /app/run_during_business_hours.sh

# Use the shell script as the entrypoint to control when the app runs
CMD ["/app/run_during_business_hours.sh"]
