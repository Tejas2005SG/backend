from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import process_pdf_and_answer, setup_vector_store
from app.db import prisma
import asyncio
import logging
import os

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the initialized Pinecone vector store
VECTOR_STORE = None

class HackrxRequest(BaseModel):
    documents: str
    questions: list[str]

class HackrxResponse(BaseModel):
    answers: list[str]

@app.on_event("startup")
async def startup_event():
    global VECTOR_STORE
    try:
        # ✅ Connect to database
        await prisma.connect()

        # ✅ Construct absolute path to the PDF file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(base_dir, "datasets", "Arogya_Sanjeevani_Policy.pdf")

        # ✅ Set up vector store (run sync code in thread)
        VECTOR_STORE = await asyncio.to_thread(setup_vector_store, pdf_path)

    except Exception as e:
        logger.error(f"Failed to set up vector store: {e}")
        raise RuntimeError(f"Application failed to start due to RAG setup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await prisma.disconnect()

@app.post("/api/v1/hackrx/run", response_model=HackrxResponse)
async def hackrx_run(data: HackrxRequest, request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key or api_key != "hackrx12345":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not VECTOR_STORE:
        raise HTTPException(status_code=503, detail="Service not ready. Vector store is still initializing.")

    try:
        # Run synchronous function in executor
        loop = asyncio.get_event_loop()
        answers = await loop.run_in_executor(
            None, 
            lambda: process_pdf_and_answer(VECTOR_STORE, data.questions)
        )

        # Save to DB
        for q, a in zip(data.questions, answers):
            await prisma.querylog.create(
                data={
                    "document": data.documents,
                    "question": q,
                    "answer": a
                }
            )

        return {"answers": answers}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
