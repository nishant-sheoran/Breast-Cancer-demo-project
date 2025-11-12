# Base image for Streamlit
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
