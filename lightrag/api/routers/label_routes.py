"""
Label management API routes for LightRAG server
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Set, Optional, Any
import logging
from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class LabelCreate(BaseModel):
    """Request model for creating a new label"""
    name: str = Field(..., description="Label name", min_length=1, max_length=100)
    description: str = Field("", description="Label description", max_length=500)
    color: str = Field("#0066cc", description="Label color in hex format", pattern="^#[0-9A-Fa-f]{6}$")

class LabelResponse(BaseModel):
    """Response model for label information"""
    name: str
    description: str
    color: str
    created_at: str
    document_count: int

class LabelAssignment(BaseModel):
    """Request model for assigning labels to a document"""
    doc_id: str = Field(..., description="Document ID")
    labels: List[str] = Field(..., description="List of labels to assign")
    file_path: str = Field("", description="Optional file path for organization")

class LabelStatistics(BaseModel):
    """Response model for label statistics"""
    total_labels: int
    total_labeled_documents: int
    labels_with_counts: Dict[str, int]

class DocumentLabelsResponse(BaseModel):
    """Response model for document labels"""
    doc_id: str
    labels: List[str]

class LabelDocumentsResponse(BaseModel):
    """Response model for documents with a specific label"""
    label_name: str
    document_ids: List[str]

class DocumentUploadWithLabels(BaseModel):
    """Request model for document upload with labels"""
    content: str = Field(..., description="Document content")
    labels: List[str] = Field(default_factory=list, description="Labels to assign")
    doc_id: Optional[str] = Field(None, description="Optional document ID")
    file_path: Optional[str] = Field(None, description="Optional file path")

class BulkLabelOperation(BaseModel):
    """Request model for bulk label operations"""
    document_ids: List[str] = Field(..., description="List of document IDs")
    labels: List[str] = Field(..., description="Labels to assign to all documents")

def create_label_routes(rag: LightRAG, api_key: Optional[str] = None) -> APIRouter:
    """Create and return the label management router"""
    router = APIRouter(prefix="/api/labels", tags=["labels"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/", response_model=Dict[str, LabelResponse])
    async def get_all_labels(_=Depends(combined_auth)):
        """Get all available labels"""
        try:
            labels = rag.get_all_labels()
            return {
                name: LabelResponse(
                    name=label.name,
                    description=label.description,
                    color=label.color,
                    created_at=label.created_at,
                    document_count=len(rag.label_manager.get_documents_by_label(name))
                )
                for name, label in labels.items()
            }
        except Exception as e:
            logger.error(f"Error getting labels: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve labels: {str(e)}"
            )

    @router.post("/", response_model=LabelResponse, status_code=status.HTTP_201_CREATED)
    async def create_label(
        label_data: LabelCreate,
        _=Depends(combined_auth)
    ):
        """Create a new label"""
        try:
            success = await rag.create_label(
                label_data.name, 
                label_data.description, 
                label_data.color
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Label '{label_data.name}' already exists"
                )
            
            # Return the created label
            labels = rag.get_all_labels()
            if label_data.name in labels:
                label = labels[label_data.name]
                return LabelResponse(
                    name=label.name,
                    description=label.description,
                    color=label.color,
                    created_at=label.created_at,
                    document_count=0
                )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Label created but could not be retrieved"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating label: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create label: {str(e)}"
            )

    @router.delete("/{label_name}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_label(
        label_name: str,
        _=Depends(combined_auth)
    ):
        """Delete a label"""
        try:
            success = await rag.delete_label(label_name)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label '{label_name}' not found"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting label: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete label: {str(e)}"
            )

    @router.get("/statistics", response_model=LabelStatistics)
    async def get_label_statistics(_=Depends(combined_auth)):
        """Get label usage statistics"""
        try:
            stats = rag.get_label_statistics()
            return LabelStatistics(**stats)
        except Exception as e:
            logger.error(f"Error getting label statistics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve statistics: {str(e)}"
            )

    @router.post("/assign", response_model=DocumentLabelsResponse)
    async def assign_labels_to_document(
        assignment: LabelAssignment,
        _=Depends(combined_auth)
    ):
        """Assign labels to a document"""
        try:
            success = await rag.assign_labels_to_document(
                assignment.doc_id,
                assignment.labels,
                assignment.file_path
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to assign labels to document '{assignment.doc_id}'"
                )
            
            # Return updated document labels
            doc_labels = rag.get_document_labels(assignment.doc_id)
            return DocumentLabelsResponse(
                doc_id=assignment.doc_id,
                labels=list(doc_labels)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error assigning labels: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to assign labels: {str(e)}"
            )

    @router.post("/assign-bulk", response_model=List[DocumentLabelsResponse])
    async def assign_labels_bulk(
        bulk_assignment: BulkLabelOperation,
        _=Depends(combined_auth)
    ):
        """Assign labels to multiple documents"""
        try:
            results = []
            failed_docs = []
            
            for doc_id in bulk_assignment.document_ids:
                try:
                    success = await rag.assign_labels_to_document(doc_id, bulk_assignment.labels)
                    if success:
                        doc_labels = rag.get_document_labels(doc_id)
                        results.append(DocumentLabelsResponse(
                            doc_id=doc_id,
                            labels=list(doc_labels)
                        ))
                    else:
                        failed_docs.append(doc_id)
                except Exception as e:
                    logger.warning(f"Failed to assign labels to {doc_id}: {e}")
                    failed_docs.append(doc_id)
            
            if failed_docs:
                logger.warning(f"Failed to assign labels to documents: {failed_docs}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk label assignment: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to assign labels: {str(e)}"
            )

    @router.get("/documents/{label_name}", response_model=LabelDocumentsResponse)
    async def get_documents_by_label(
        label_name: str,
        _=Depends(combined_auth)
    ):
        """Get all documents with a specific label"""
        try:
            doc_ids = rag.get_documents_by_label(label_name)
            return LabelDocumentsResponse(
                label_name=label_name,
                document_ids=list(doc_ids)
            )
        except Exception as e:
            logger.error(f"Error getting documents by label: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve documents: {str(e)}"
            )

    @router.get("/document/{doc_id}", response_model=DocumentLabelsResponse)
    async def get_document_labels(
        doc_id: str,
        _=Depends(combined_auth)
    ):
        """Get all labels for a specific document"""
        try:
            labels = rag.get_document_labels(doc_id)
            return DocumentLabelsResponse(
                doc_id=doc_id,
                labels=list(labels)
            )
        except Exception as e:
            logger.error(f"Error getting document labels: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve document labels: {str(e)}"
            )

    @router.post("/upload-with-labels", response_model=Dict[str, Any])
    async def upload_document_with_labels(
        document: DocumentUploadWithLabels,
        _=Depends(combined_auth)
    ):
        """Upload a document with labels"""
        try:
            # Insert document with labels
            track_id = rag.insert_with_labels(
                input=document.content,
                labels=document.labels,
                ids=[document.doc_id] if document.doc_id else None,
                file_paths=[document.file_path] if document.file_path else None
            )
            
            return {
                "status": "success",
                "track_id": track_id,
                "doc_id": document.doc_id,
                "labels": document.labels,
                "message": f"Document uploaded with labels: {document.labels}"
            }
            
        except Exception as e:
            logger.error(f"Error uploading document with labels: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload document: {str(e)}"
            )

    return router