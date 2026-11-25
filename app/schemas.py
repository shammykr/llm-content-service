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
