FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies first, then ODBC
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https \
    gcc \
    g++ \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft repository and install ODBC driver first
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install unixODBC after Microsoft ODBC driver to avoid conflicts
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify ODBC installation
RUN odbcinst -j

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Test pyodbc import
RUN python -c "import pyodbc; print('pyodbc imported successfully')"

# Copy application code
COPY . .

# Set environment variables for ODBC
ENV ODBCSYSINI=/etc
ENV ODBCINI=/etc/odbc.ini

EXPOSE 8008

CMD ["python", "serversse.py"]