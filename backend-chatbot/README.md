# Backend Chatbot FastAPI

API REST para chatbot usando FastAPI y MongoDB.

## Ejecutar localmente

1. Clonar el repo
2. Crear entorno virtual e instalar dependencias

```bash
pip install -r requirements.txt
```

3. Ejecutar con:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

4. Cambia `MONGO_URI` en `main.py` por tu conexión MongoDB si usas local.

## Deploy en Railway

- Añade la variable de entorno `PORT` (Railway la asigna automáticamente).
- Configura el `MONGO_URI` con tu MongoDB de Railway.
- Ejecuta con `uvicorn main:app --host 0.0.0.0 --port $PORT`
