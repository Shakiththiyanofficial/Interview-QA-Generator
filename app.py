from fastapi import FastAPI, Form, Request, Response, File, UploadFile, HTTPException, status
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    # Validate file type
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    
    pdf_filename = os.path.join(base_folder, pdf_file.filename)
    
    # Save the uploaded file
    async with aiofiles.open(pdf_filename, 'wb') as f:
        content = await pdf_file.read()
        await f.write(content)
    
    return JSONResponse({
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_filename": pdf_filename,
        "filename": pdf_file.filename
    })

def get_csv(file_path):
    try:
        answer_generation_chain, filtered_questions_list = llm_pipeline(file_path)
        base_folder = 'static/output/'
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        
        output_file = os.path.join(base_folder, "QA.csv")
        
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Question", "Answer"])  # Writing the header row
            
            for question in filtered_questions_list:
                print("Question: ", question)
                answer = answer_generation_chain.run(question)
                print("Answer: ", answer)
                print("." * 40 + "\n\n")
                
                # Save answer to CSV file
                csv_writer.writerow([question, answer])
        
        return output_file
    except Exception as e:
        print(f"Error in get_csv: {str(e)}")
        raise e

@app.post("/analyze")
async def analyze_pdf(request: Request, pdf_filename: str = Form(...)):
    try:
        if not os.path.exists(pdf_filename):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        output_file = get_csv(pdf_filename)
        
        return JSONResponse({
            "status": "success",
            "message": "Analysis completed successfully",
            "output_file": output_file
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }, status_code=500)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"static/output/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv'
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000 ,reload=True)