"""
API routes for document management (upload, list, delete, processing status).
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form
from sqlalchemy.orm import Session
import uuid
import os
from pathlib import Path
import json

from ...core.database import get_db
from ...models.document import Document as DocumentModel
from ...schemas.document import (
    DocumentResponse, 
    DocumentListResponse, 
    DocumentUploadResponse,
    DocumentProcessingResponse,
    DocumentCreate,
    ProcessingStatus
)
from ...services.rag_service import rag_service
from ...services.document_processor import document_processor
from ..dependencies import validate_file_upload

logger = logging.getLogger(__name__)


with open('/home/ubuntu/s2orc/s2orc/20251212_123706_00007_5x9kq_0b674d39-1a44-4420-a64b-f2f5f1676292') as f:
    idx = 0
    for line in f:
        j_content = json.loads(line)
        #pprint(j_content)
        print(j_content['externalids']['doi'])
        if j_content['externalids']['doi']:
            #print(works.doi(j_content['externalids']['doi'])['published']['date-parts'])
            print(j_content['content']['annotations'].keys())
            #pprint(j_content)
            content = j_content['content']['text']
            authorfirstnames = json.loads(j_content['content']['annotations']['authorfirstname'])
            authorlastnames = json.loads(j_content['content']['annotations']['authorlastname'])
            authoraffiliations = json.loads(j_content['content']['annotations']['authoraffiliation'])
            year = works.doi(j_content['externalids']['doi'])['published']['date-parts']

            for order,fname in enumerate(authorfirstnames):
                lname = authorlastnames[order]
                print(content[fname['start']:fname['end']],content[lname['start']:lname['end']])
            print(j_content['content']['annotations'].keys())
            metadata = {'venue':'whatever'}
            multipart_data = MultipartEncoder( fields={'file': ('test.txt', content.encode('utf-8'), 'text/plain'), 'document_metadata': json.dumps(metadata)>
            #response=requests.post(url, data=multipart_data, headers={'Content-Type': multipart_data.content_type})
            #print(response)
        else:
            print(j_content)
#        idx+=1
#        if idx>100:
#            break
#        break

async def upload_document(
    background_tasks: BackgroundTasks,
    document_metadata: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document for processing.
    
    - **file**: Document file (PDF, DOCX, TXT, MD, CSV)
    
    Returns uploaded document info and starts background processing.
    """
    try:
        # Validate file
        file_content = await file.read()
        validate_file_upload(len(file_content), file.filename)
        
        # Save file to disk
        file_path = await document_processor.save_uploaded_file(
            file_content, file.filename
        )
        
        # Create document record
        document_data = DocumentCreate(
            filename=Path(file_path).name,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(file_content),
            file_type=Path(file.filename).suffix.lower(),
            mime_type=file.content_type
        )
        
        document_model = DocumentModel(**document_data.dict())
        db.add(document_model)
        db.commit()
        db.refresh(document_model)
        
        # Start background processing
        background_tasks.add_task(
            rag_service.process_document_pipeline,
            document_model,
            document_metadata,
            db
        )
        
        upload_id = str(uuid.uuid4())
        
        logger.info(f"Document uploaded: {document_model.id} - {file.filename}")
        
        return DocumentUploadResponse(
            message="Document uploaded successfully and processing started",
            document=DocumentResponse.from_orm(document_model),
            upload_id=upload_id
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

