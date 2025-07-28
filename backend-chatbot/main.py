from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import motor.motor_asyncio
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Inicializar la app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión MongoDB
MONGO_URI = "mongodb://mongo:JqIXnWRvbqNLobljNLGYFcloiKymZfbf@mongodb.railway.internal:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.IA
formularios = db.formularios
pendientes = db.aprendizaje_formulario

# Base de datos de preguntas frecuentes
preguntas = [
    "¿Cuántas horas necesito de prácticas laborales?",
    "¿Cuáles son los requisitos para convalidar prácticas?",
    "¿Dónde puedo hacer vinculación?",
    "¿Qué hago si ya tengo tutor para mis prácticas?",
    "¿Puedo hacer prácticas con familiares?",
    "¿Qué documentos debo enviar para el registro de prácticas?",
    "¿Cuánto tiempo de validez tienen los certificados de prácticas?",
]

respuestas = [
    "Necesitas 240 horas de prácticas laborales y 96 de servicio comunitario (Tecnología Superior).",
    "Debes enviar un correo con el formulario FCP-001A y certificado de actividad. La subdirección lo revisa y designa tutor.",
    "Puedes hacer vinculación en entidades públicas o privadas que certifiquen las actividades y horas realizadas.",
    "Debes escribirle directamente a tu tutor y seguir el procedimiento de registro de prácticas.",
    "No puedes hacer prácticas en emprendimientos u organizaciones de familiares o compañeros de la EPN.",
    "Debes subir un archivo PDF con la Carta de Aceptación y la Solicitud de Prácticas en el formulario de registro.",
    "Tienen una validez de 6 meses desde la fecha de finalización de las prácticas o de emisión del certificado."
]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preguntas)

# Modelo de formulario
class Formulario(BaseModel):
    nombre: str
    cedula: str
    carrera: str
    modalidad: str
    tipo_actividad: str
    descripcion: str

# Modelo de mensaje
class Pregunta(BaseModel):
    texto: str

@app.get("/")
def root():
    return {"message": "Servidor de predicción de prácticas activo"}

@app.post("/formulario/registro")
async def registrar_formulario(data: Formulario):
    form_dict = data.dict()
    result = await formularios.insert_one(form_dict)
    form_dict["_id"] = str(result.inserted_id)
    return {"message": "Formulario recibido", "data": form_dict}

@app.post("/formulario/pregunta")
async def responder_pregunta(pregunta: Pregunta):
    texto = pregunta.texto
    query_vec = vectorizer.transform([texto])
    similitudes = (X * query_vec.T).toarray().flatten()

    if np.max(similitudes) < 0.2:
        await pendientes.insert_one({"pregunta": texto})
        return {
            "respuesta": "Lo siento, no tengo información suficiente sobre esa pregunta. Será enviada para revisión.",
            "necesita_aprendizaje": True
        }

    idx = np.argmax(similitudes)
    return {
        "respuesta": respuestas[idx],
        "necesita_aprendizaje": False
    }

@app.get("/formularios")
async def listar_formularios():
    lista = []
    async for form in formularios.find():
        form["_id"] = str(form["_id"])
        lista.append(form)
    return lista

@app.delete("/formulario/{id_form}")
async def eliminar_formulario(id_form: str):
    res = await formularios.delete_one({"_id": ObjectId(id_form)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Formulario no encontrado")
    return {"message": "Formulario eliminado"}
