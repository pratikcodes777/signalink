# Use the Python 3.10 image as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY app.py .
COPY requirements.txt .

# Install application dependencies
RUN pip install -r requirements.txt

# Specify the command to run the application
CMD ["python", "app.py"]



