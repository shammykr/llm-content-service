from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    max_sentences: int = Field(default=3, ge=1, le=20)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    target_language: str = Field(min_length=2, examples=["French"])


class Candidate(BaseModel):
    id: str
    title: str
    description: str | None = None


class RecommendRequest(BaseModel):
    content: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=50)


class Recommendation(BaseModel):
    id: str
    reason: str = ""


class TextResult(BaseModel):
    result: str
    cached: bool


class RecommendResult(BaseModel):
    recommendations: list[Recommendation]
    cached: bool


class IngestDocument(BaseModel):
    id: str | None = Field(default=None, max_length=128)
    text: str = Field(min_length=1)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1, max_length=1000)


class IngestAck(BaseModel):
    id: str
    status: str
    language: str
    token_count: int
    sentence_count: int


class IngestResponse(BaseModel):
    queued: int
    documents: list[IngestAck]


class StreamIngestResponse(BaseModel):
    queued: int
    rejected: list[dict]
    documents: list[IngestAck]


class IngestStatus(IngestAck):
    summary: str | None = None
    error: str | None = None
