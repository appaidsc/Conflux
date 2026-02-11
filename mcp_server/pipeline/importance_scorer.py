"""
Pipeline - Importance Scorer

Calculates importance score (0.0-3.0) based on structural, hierarchy, labels, and content signals.
"""

from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass

from mcp_server.pipeline.analyzer import AnalysisResult


# Labels that indicate important content
IMPORTANT_LABELS = {
    "overview", "guide", "tutorial", "architecture",
    "getting-started", "reference", "api", "howto",
    "best-practices", "documentation", "runbook",
    "setup", "installation", "configuration",
}


@dataclass
class ImportanceResult:
    """Importance scoring result."""
    score: float
    classification: Literal["high", "medium", "low"]
    signals: Dict[str, float]
    matched_labels: List[str]


class ImportanceScorer:
    """Calculates importance score based on multiple signals."""
    
    def score(
        self,
        children_count: int,
        depth: int,
        labels: List[str],
        analysis: AnalysisResult,
    ) -> ImportanceResult:
        """
        Calculate importance score.
        
        Score = Structural + Hierarchy + Labels + Content (max 3.0)
        
        Args:
            children_count: Number of child pages
            depth: Depth in page hierarchy (0 = root)
            labels: Page labels/tags
            analysis: Content analysis result
            
        Returns:
            ImportanceResult with score and breakdown
        """
        signals = {}
        
        # STRUCTURAL (max 1.5)
        # Hub pages with many children are important
        structural = min(1.5,
            children_count * 0.1 +           # 10 children = 1.0
            analysis.internal_links_count * 0.05  # 10 links = 0.5
        )
        signals["structural"] = round(structural, 2)
        
        # HIERARCHY (max 0.5)
        # Top-level pages are more important
        if depth <= 1:
            hierarchy = 0.5
        elif depth == 2:
            hierarchy = 0.3
        else:
            hierarchy = 0.0
        signals["hierarchy"] = hierarchy
        
        # LABELS (max 1.0)
        lowercase_labels = [l.lower() for l in labels]
        matched = [l for l in lowercase_labels if l in IMPORTANT_LABELS]
        label_score = min(1.0, len(matched) * 0.25)
        signals["labels"] = label_score
        
        # CONTENT (max 0.8)
        content = 0.0
        if analysis.has_code:
            content += 0.3
        if analysis.has_images:
            content += 0.3
        if analysis.word_count > 500:
            content += 0.2
        signals["content"] = min(0.8, content)
        
        # Total score
        total = sum(signals.values())
        
        # Classification
        if total >= 2.0:
            classification = "high"
        elif total >= 1.0:
            classification = "medium"
        else:
            classification = "low"
        
        return ImportanceResult(
            score=round(total, 2),
            classification=classification,
            signals=signals,
            matched_labels=matched,
        )
