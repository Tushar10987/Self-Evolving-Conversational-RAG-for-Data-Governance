"""
HyDE (Hypothesis-driven Document Enhancement) implementation.
Generates candidate answers before retrieval to improve document matching.
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging

# LLM components
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import hyde_config, llm_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HypothesisResult:
    """Result from hypothesis generation."""
    hypotheses: List[str]
    generation_time: float
    model_used: str

class HyDEGenerator:
    """Hypothesis-driven Document Enhancement generator."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or llm_config.LOCAL_MODEL
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model if using local
        if llm_config.USE_LOCAL:
            self._load_model()
    
    def _load_model(self):
        """Load the local language model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _generate_with_local_model(self, prompt: str, max_length: int = None) -> str:
        """Generate text using local model."""
        if self.model is None or self.tokenizer is None:
            return ""
        
        max_length = max_length or llm_config.LOCAL_MODEL_MAX_LENGTH
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length//2)
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=llm_config.LOCAL_MODEL_TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            return ""
    
    def _generate_with_remote_api(self, prompt: str) -> str:
        """Generate text using remote API (placeholder for future implementation)."""
        # This would integrate with OpenAI, Anthropic, or other APIs
        # For now, return a placeholder response
        logger.warning("Remote API not configured, using placeholder response")
        return "This is a placeholder response from remote API."
    
    def _create_hypothesis_prompt(self, query: str) -> str:
        """Create a prompt for hypothesis generation."""
        prompt_templates = [
            f"Given the question: '{query}', what would be a likely answer? Provide a brief response that might be found in a document:",
            f"Based on the query '{query}', generate a potential answer that could be retrieved from a knowledge base:",
            f"For the question '{query}', what information would be most relevant? Write a short answer:",
            f"Imagine you're searching for information about '{query}'. What kind of answer would you expect to find?",
            f"Given '{query}', what would be a reasonable response that could be found in documentation?"
        ]
        
        return random.choice(prompt_templates)
    
    def generate_hypotheses(self, query: str, num_hypotheses: int = None) -> HypothesisResult:
        """Generate multiple hypotheses for a given query."""
        start_time = time.time()
        
        num_hypotheses = num_hypotheses or hyde_config.HYPOTHESIS_COUNT
        hypotheses = []
        
        for i in range(num_hypotheses):
            # Create different prompts for variety
            prompt = self._create_hypothesis_prompt(query)
            
            # Generate hypothesis
            if llm_config.USE_LOCAL and self.model is not None:
                hypothesis = self._generate_with_local_model(
                    prompt, 
                    max_length=hyde_config.HYPOTHESIS_LENGTH
                )
            else:
                hypothesis = self._generate_with_remote_api(prompt)
            
            # Clean and validate hypothesis
            if hypothesis and len(hypothesis.strip()) > 10:
                hypotheses.append(hypothesis.strip())
        
        generation_time = time.time() - start_time
        model_used = self.model_name if llm_config.USE_LOCAL else "remote_api"
        
        logger.info(f"Generated {len(hypotheses)} hypotheses in {generation_time:.2f}s")
        
        return HypothesisResult(
            hypotheses=hypotheses,
            generation_time=generation_time,
            model_used=model_used
        )
    
    def enhance_query(self, query: str, use_hypotheses: bool = True) -> Dict[str, Any]:
        """Enhance a query with hypotheses for better retrieval."""
        if not use_hypotheses or not hyde_config.ENABLED:
            return {
                "original_query": query,
                "enhanced_queries": [query],
                "hypotheses": [],
                "enhancement_weight": 0.0
            }
        
        # Generate hypotheses
        hypothesis_result = self.generate_hypotheses(query)
        
        # Create enhanced queries
        enhanced_queries = [query]  # Always include original query
        
        for hypothesis in hypothesis_result.hypotheses:
            # Combine original query with hypothesis
            enhanced_query = f"{query} {hypothesis}"
            enhanced_queries.append(enhanced_query)
        
        return {
            "original_query": query,
            "enhanced_queries": enhanced_queries,
            "hypotheses": hypothesis_result.hypotheses,
            "enhancement_weight": hyde_config.ENHANCEMENT_WEIGHT,
            "generation_time": hypothesis_result.generation_time,
            "model_used": hypothesis_result.model_used
        }

class HyDERetriever:
    """Retriever that uses HyDE for enhanced document retrieval."""
    
    def __init__(self, base_retriever, hyde_generator: HyDEGenerator = None):
        self.base_retriever = base_retriever
        self.hyde_generator = hyde_generator or HyDEGenerator()
    
    def retrieve_with_hyde(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Retrieve documents using HyDE enhancement."""
        start_time = time.time()
        
        # Generate enhanced queries
        enhancement_result = self.hyde_generator.enhance_query(query)
        
        # Retrieve with original query
        original_result = self.base_retriever.retrieve(query, top_k)
        
        # Retrieve with enhanced queries
        enhanced_results = []
        for enhanced_query in enhancement_result["enhanced_queries"][1:]:  # Skip original
            result = self.base_retriever.retrieve(enhanced_query, top_k)
            enhanced_results.append(result)
        
        # Combine results
        combined_docs = self._combine_results(original_result, enhanced_results)
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "original_result": original_result,
            "enhanced_queries": enhancement_result["enhanced_queries"],
            "hypotheses": enhancement_result["hypotheses"],
            "combined_documents": combined_docs,
            "total_time": total_time,
            "enhancement_weight": enhancement_result["enhancement_weight"]
        }
    
    def _combine_results(self, original_result, enhanced_results: List) -> List:
        """Combine results from original and enhanced queries."""
        # Simple combination strategy: take unique documents with highest scores
        doc_scores = {}
        
        # Add original results
        for doc, score in zip(original_result.documents, original_result.scores):
            doc_id = doc.page_content[:100]
            if doc_id not in doc_scores or score > doc_scores[doc_id]["score"]:
                doc_scores[doc_id] = {"doc": doc, "score": score}
        
        # Add enhanced results with weight adjustment
        for enhanced_result in enhanced_results:
            for doc, score in zip(enhanced_result.documents, enhanced_result.scores):
                doc_id = doc.page_content[:100]
                adjusted_score = score * hyde_config.ENHANCEMENT_WEIGHT
                
                if doc_id not in doc_scores or adjusted_score > doc_scores[doc_id]["score"]:
                    doc_scores[doc_id] = {"doc": doc, "score": adjusted_score}
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

def create_hyde_generator() -> HyDEGenerator:
    """Create a HyDE generator instance."""
    return HyDEGenerator()

def create_hyde_retriever(base_retriever) -> HyDERetriever:
    """Create a HyDE-enhanced retriever."""
    hyde_generator = create_hyde_generator()
    return HyDERetriever(base_retriever, hyde_generator)

if __name__ == "__main__":
    # Test HyDE generation
    hyde_gen = create_hyde_generator()
    
    test_queries = [
        "Who owns the customer_data table?",
        "What tables contain sensitive information?",
        "Show me data lineage for financial data",
        "Which departments have access to user data?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        enhancement = hyde_gen.enhance_query(query)
        print(f"Original query: {enhancement['original_query']}")
        print(f"Hypotheses: {enhancement['hypotheses']}")
        print(f"Enhanced queries: {len(enhancement['enhanced_queries'])}")
        print(f"Generation time: {enhancement['generation_time']:.2f}s") 