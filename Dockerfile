# Usa una imagen base ligera
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los archivos necesarios antes de instalar dependencias
COPY requirements.txt .

# Instala solo las librerías especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que corre la aplicación
EXPOSE 8080

# Define el comando de ejecución de la aplicación
CMD ["python3", "application.py"]
