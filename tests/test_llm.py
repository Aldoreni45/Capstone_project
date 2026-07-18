import pytest
from pydantic import ValidationError
from pydantic_models.responses import StructuredAnswer
from pydantic_models.citation import Citation

def test_structured_answer_validation():
    """Checks model validators for the StructuredAnswer Pydantic schema."""
    # Valid model
    ans = StructuredAnswer(
        answer="The transformer architecture uses attention mechanism.",
        citations=[
            Citation(
                title="Attention Is All You Need",
                page=3,
                chunk_id="att_p3_c0",
                source="attention.pdf",
                passage="The transformer architecture relies entirely on self-attention."
            )
        ],
        confidence_score=0.95
    )
    assert ans.confidence_score == 0.95
    
    # Invalid confidence score (under 0.0)
    with pytest.raises(ValidationError):
        StructuredAnswer(
            answer="Answer details",
            citations=[],
            confidence_score=-0.1
        )
        
    # Invalid confidence score (above 1.0)
    with pytest.raises(ValidationError):
        StructuredAnswer(
            answer="Answer details",
            citations=[],
            confidence_score=1.1
        )
