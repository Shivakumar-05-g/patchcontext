import re
import logging
from src import config
from src.citations import extract_citations, validate_citations
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class HallucinationGuard:
    def __init__(self, use_local_nli=True):
        self.use_local_nli = use_local_nli
        self.nli_pipeline = None

        logger.info("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Sentence embedding model loaded.")
            
        if self.use_local_nli:
            try:
                from transformers import pipeline
                logger.info(f"Initializing local NLI model: {config.NLI_MODEL_NAME}...")

                self.nli_pipeline = pipeline(
                    "text-classification",
                    model=config.NLI_MODEL_NAME,
                    device=-1
                )
                logger.info("Local NLI model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Could not load local NLI model ({e}). "
                    "Hallucination guard will fall back to LLM-based NLI evaluation."
                )
                self.nli_pipeline = None

    def split_into_sentences(self, text):
        """
        Split generated answer into factual claims suitable for NLI verification.

        This function removes:
        - Boilerplate phrases
        - Citation-only lines
        - Very short fragments
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        cleaned = []

        skip_patterns = [
            r"^yes[, ]",
            r"^no[, ]",
            r"^according to",
            r"^based on",
            r"^from the retrieved",
            r"^retrieved context",
            r"^\[?issue\s*#?\d+\]?$",
            r"^\[?pr\s*#?\d+\]?$",
            r"^\[?commit\s+[0-9a-f]{6,40}\]?$",
            r"^https?://",
        ]

        for sentence in sentences:
            s = sentence.strip()

            if len(s) < 20:
                continue

            # Skip citation-only sentences
            should_skip = False
            for pattern in skip_patterns:
                if re.match(pattern, s, re.IGNORECASE):
                    should_skip = True
                    break

            if should_skip:
                continue

            cleaned.append(s)

        return cleaned

    def evaluate_statement_llm_nli(self, premise, hypothesis):
        """Fallback LLM-based NLI evaluation using ChatGroq."""
        try:
            from src.rag_chain import get_llm
            llm = get_llm()
            
            prompt = (
                "You are an NLI (Natural Language Inference) auditor. Your task is to evaluate "
                "whether a claim (Hypothesis) is supported, contradicted, or neutral relative to the "
                "provided Context (Premise).\n\n"
                "=== CONTEXT (PREMISE) ===\n"
                f"{premise}\n"
                "=========================\n\n"
                "=== CLAIM (HYPOTHESIS) ===\n"
                f"{hypothesis}\n"
                "==========================\n\n"
                "Determine the relationship and respond in exactly this format:\n"
                "LABEL: [entailment | contradiction | neutral]\n"
                "REASON: [Brief 1-sentence reason for classification]\n\n"
                "Definitions:\n"
                "- entailment: The context directly supports or proves the claim.\n"
                "- contradiction: The context directly contradicts the claim.\n"
                "- neutral: The context does not contain enough information to prove or disprove the claim."
            )
            
            response = llm.invoke(prompt)
            content = response.content.lower()
            
            # Parse classification label
            if "entailment" in content:
                label = "entailment"
            elif "contradiction" in content:
                label = "contradiction"
            else:
                label = "neutral"
                
            return label, 1.0
        except Exception as e:
            logger.error(f"Error in LLM-based NLI fallback: {e}")
            return "neutral", 0.0

    def evaluate_statement_local_nli(self, premise, hypothesis):
        """
        Evaluate a statement using the CrossEncoder NLI model.
        """
        if self.nli_pipeline is None:
            return "neutral", 0.0

        try:
            res = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
            label = res["label"].upper()
            score = res["score"]

            if "ENTAIL" in label:
                mapped_label = "entailment"
            elif "CONTRADICT" in label or "LABEL_0" in label:
                mapped_label = "contradiction"
            elif "NEUTRAL" in label or "LABEL_2" in label:
                mapped_label = "neutral"
            else:
                mapped_label = "neutral"

            return mapped_label, score

        except Exception as e:
            logger.error(f"Local NLI evaluation failed: {e}")
            return "neutral", 0.0

    def get_best_matching_sentence(self, claim, document):
        """
        Find the sentence in the retrieved document
        that is most semantically similar to the claim.

        Running NLI on this sentence instead of the
        entire document greatly improves entailment.
        """
        # Split document into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+',
            document
        )

        # Remove tiny fragments
        sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 20 and len(s.split()) > 5
        ]

        if not sentences:
            return document[:512]

        # Encode claim
        claim_embedding = self.embedding_model.encode(
            [claim],
            convert_to_numpy=True
        )

        # Encode sentences
        sentence_embeddings = self.embedding_model.encode(
            sentences,
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            claim_embedding,
            sentence_embeddings
        )[0]

        best_index = int(np.argmax(similarities))

        return sentences[best_index]

    def check_hallucinations(self, answer, retrieved_docs):
        """
        Check generated answer for unsupported statements and invalid citations.

        Improvements:
        - Compare each statement against EACH retrieved document separately.
        - Keep the strongest entailment result.
        - Do not automatically treat Neutral as hallucination.
        - Preserve existing return format.
        """
        logger.info("=== Running Hallucination Guard ===")

        # 1. Citation Validation
        citations = extract_citations(answer)
        validated_citations, invalid_citations = validate_citations(
            citations,
            retrieved_docs
        )

        # 2. Split answer into statements
        sentences = self.split_into_sentences(answer)
        logger.info(f"Extracted sentences: {sentences}")

        statement_reports = []
        has_hallucinations = False

        # 3. Verify every statement
        for sent in sentences:
            # Ignore obvious boilerplate
            sent_clean = re.sub(
                r"^according to the retrieved context[:,]?\s*",
                "",
                sent,
                flags=re.IGNORECASE
            )
            sent_clean = re.sub(
                r"^based on the retrieved context[:,]?\s*",
                "",
                sent_clean,
                flags=re.IGNORECASE
            )

            if len(sent_clean.strip()) < 10:
                continue

            best_label = "neutral"
            best_score = 0.0
            best_doc = None
            method = "local_nli" if self.nli_pipeline else "llm_nli"

            # Compare against every retrieved document
            for doc in retrieved_docs:
                premise = self.get_best_matching_sentence(
                    sent_clean,
                    doc.page_content
                )

                if self.nli_pipeline:
                    label, score = self.evaluate_statement_local_nli(
                        premise,
                        sent_clean
                    )
                else:
                    label, score = self.evaluate_statement_llm_nli(
                        premise,
                        sent_clean
                    )

                # Highest entailment always wins
                if label == "entailment" and score > best_score:
                    best_label = label
                    best_score = score
                    best_doc = doc
                # Otherwise keep highest confidence
                elif (
                    best_label != "entailment"
                    and score > best_score
                ):
                    best_label = label
                    best_score = score
                    best_doc = doc

            # Decide whether to flag
            flagged = False

            # Contradictions are always suspicious
            if best_label == "contradiction":
                flagged = True
                has_hallucinations = True
            # Only flag neutral if confidence is extremely high
            elif (
                best_label == "neutral"
                and best_score >= 0.98
            ):
                flagged = True
                has_hallucinations = True

            statement_reports.append(
                {
                    "statement": sent,
                    "nli_label": best_label,
                    "confidence": round(best_score, 3),
                    "flagged": flagged,
                    "method": method,
                    "matched_source": (
                        best_doc.metadata
                        if best_doc
                        else None
                    )
                }
            )

        is_safe = (
            not has_hallucinations
            and len(invalid_citations) == 0
        )

        logger.info(
            f"Hallucination Guard completed. "
            f"Statements Checked: {len(statement_reports)}, "
            f"Invalid Citations: {len(invalid_citations)}, "
            f"Safe: {is_safe}"
        )

        return {
            "is_safe": is_safe,
            "statement_reports": statement_reports,
            "invalid_citations": invalid_citations,
            "validated_citations": validated_citations
        }