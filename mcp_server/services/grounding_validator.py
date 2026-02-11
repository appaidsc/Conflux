"""
Services - Grounding Validator

NLI-based claim verification to detect hallucinations.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GroundingResult:
    """Grounding validation result."""
    score: float  # 0.0 - 1.0
    valid_claims: int
    total_claims: int
    ungrounded_claims: List[str]


class GroundingValidator:
    """Validates that generated answers are grounded in source documents."""
    
    async def validate(
        self,
        answer: str,
        sources: List[str],
        llm_provider=None,
    ) -> GroundingResult:
        """
        Validate answer claims against source documents.
        
        Args:
            answer: Generated answer text
            sources: List of source document texts
            llm_provider: Optional LLM for validation
            
        Returns:
            GroundingResult with grounding score
        """
        # Simple keyword-based validation (fast)
        # For production, use NLI model or LLM
        
        claims = self._extract_claims(answer)
        source_text = " ".join(sources).lower()
        
        valid_claims = 0
        ungrounded = []
        
        for claim in claims:
            # Check if key claim words appear in sources
            claim_words = set(
                word.lower()
                for word in claim.split()
                if len(word) > 3
            )
            
            source_words = set(source_text.split())
            overlap = len(claim_words & source_words) / max(len(claim_words), 1)
            
            if overlap >= 0.5:
                valid_claims += 1
            else:
                ungrounded.append(claim)
        
        total = len(claims) if claims else 1
        score = valid_claims / total
        
        return GroundingResult(
            score=score,
            valid_claims=valid_claims,
            total_claims=len(claims),
            ungrounded_claims=ungrounded,
        )
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        # Filter to factual statements (simple heuristic)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Skip questions and hedged statements
                if not sentence.endswith("?"):
                    if not any(hedge in sentence.lower() for hedge in [
                        "might", "maybe", "perhaps", "possibly",
                        "i think", "i believe", "it seems"
                    ]):
                        claims.append(sentence)
        
        return claims
    
    async def validate_with_llm(
        self,
        answer: str,
        sources: List[str],
        llm_provider,
    ) -> GroundingResult:
        """
        LLM-based grounding validation (more accurate but slower).
        """
        claims = self._extract_claims(answer)
        source_text = "\n\n".join(sources[:3])[:4000]
        
        prompt = f"""Verify if the following claims are supported by the source documents.
For each claim, respond with "SUPPORTED" or "NOT_SUPPORTED".

Source Documents:
{source_text}

Claims to verify:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(claims[:5]))}

For each claim (1-{min(5, len(claims))}), state if SUPPORTED or NOT_SUPPORTED:"""
        
        try:
            response = await llm_provider.complete(prompt, max_tokens=200)
            
            # Parse response
            supported_count = response.upper().count("SUPPORTED")
            not_supported = response.upper().count("NOT_SUPPORTED")
            
            # Adjust for double counting
            supported_count = supported_count - not_supported
            
            total = min(5, len(claims)) if claims else 1
            score = max(0, supported_count) / total
            
            return GroundingResult(
                score=min(1.0, score),
                valid_claims=max(0, supported_count),
                total_claims=len(claims),
                ungrounded_claims=[],  # Would need more parsing
            )
        except Exception:
            # Fall back to simple validation
            return await self.validate(answer, sources)
