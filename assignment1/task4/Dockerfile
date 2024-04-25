# Use an official Python runtime as a parent image
FROM python:3.8.10

# Set the working directory in the container
WORKDIR ~/dataengineering2/assignment1/task4/

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Define environment variable
ENV PULSAR_BROKER=pulsar://pulsar:6650

# Run script when the container launches
CMD ["python", "script.py"]
