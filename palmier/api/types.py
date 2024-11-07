from pydantic import BaseModel
from enum import Enum
from typing import Optional


class Status(Enum):
    """Status of an indexing job"""

    ACCEPTED = "accepted"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryRequest(BaseModel):
    """Request body for /query endpoint"""

    # Github repository in format "owner/repo"
    repo: str

    query: str

    # lightrag mode: naive | local | global | hybrid
    mode: Optional[str] = "hybrid"

    # return retrieved context instead of generated answer
    only_need_context: Optional[bool] = False

    # response format
    response_type: Optional[str] = "Multiple Paragraphs"

    # top k retrieval
    top_k: Optional[int] = 60


class IndexRequest(BaseModel):
    """Request body for /index endpoint"""

    # Github repository in format "owner/repo"
    repo: str

    # Repository branch - currently only one branch per repo is supported
    branch: Optional[str] = "main"


class Response(BaseModel):
    """Response body for all endpoints"""

    status: str
    data: Optional[str] = None
    message: Optional[str] = None
