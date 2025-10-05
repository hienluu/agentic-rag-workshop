"""
RAG Test Set Generator for Technical Conference Content

This script generates synthetic question-answer pairs from conference content
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add parent directory to path to import rag_llm module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_llm import RagLLM, get_llm_provider
from models import CrawlPageResult



@dataclass
class QuestionAnswer:
    """Represents a single Q&A pair with metadata"""
    question_id: str
    question: str
    answer: str
    source_chunk_id: str
    source_text: str
    difficulty: str  # easy, medium, hard
    question_type: str  # factual, conceptual, procedural, comparative, multi_hop
    session_info: Dict[str, str]  # speaker, title, topic
    confidence_score: float
    human_validated: bool = False


class ConferenceTestSetGenerator:
    """Generate synthetic test sets from conference content"""
    
    def __init__(self, llm: RagLLM):
        self.llm = llm
    
    def generate_questions_from_document(
        self, 
        document_text: str, 
        document_id: str,
        session_info: Dict[str, str],
        num_easy: int = 2,
        num_medium: int = 2,
        num_hard: int = 1
    ) -> List[QuestionAnswer]:
        """
        Generate multiple questions from a single content chunk
        
        Args:
            chunk_text: The text content from conference transcript
            chunk_id: Unique identifier for this chunk
            session_info: Dict with keys: speaker, title, topic
            num_easy/medium/hard: Number of questions per difficulty
        
        Returns:
            List of QuestionAnswer objects
        """
        questions = []
        
        # Generate easy questions (factual)
        if num_easy > 0:
            easy_questions = self._generate_factual_questions(
                document_text, document_id, session_info, num_easy
            )
            questions.extend(easy_questions)
        
        # Generate medium questions (conceptual)
        if num_medium > 0:
            medium_questions = self._generate_conceptual_questions(
                document_text, document_id, session_info, num_medium
            )
            questions.extend(medium_questions)
        
        # Generate hard questions (multi-hop/application)
        if num_hard > 0:
            hard_questions = self._generate_complex_questions(
                document_text, document_id, session_info, num_hard
            )
            questions.extend(hard_questions)
        
        return questions
    
    def _generate_factual_questions(
        self, 
        document_text: str, 
        document_id: str,
        session_info: Dict[str, str],
        num_questions: int
    ) -> List[QuestionAnswer]:
        """Generate factual/easy questions"""
        
        prompt = f"""You are creating evaluation questions for a RAG (Retrieval Augmented Generation) system based on technical conference content.


TRANSCRIPT SECTION:
{document_text}

TASK: Generate {num_questions} FACTUAL questions that can be answered directly from this transcript section.

REQUIREMENTS FOR FACTUAL QUESTIONS:
1. Focus on specific information: tools, technologies, metrics, names, dates
2. Questions should have clear, unambiguous answers found directly in the text
3. Appropriate for testing retrieval accuracy
4. Relevant to senior engineers and technical leaders
5. Should NOT require inference or external knowledge

EXAMPLES OF GOOD FACTUAL QUESTIONS:
- "What database technology did [Speaker] recommend for high-throughput scenarios?"
- "According to the presentation, what was the latency improvement achieved?"
- "What are the three main components of the architecture discussed?"

FORMAT YOUR RESPONSE AS JSON:
{{
  "questions": [
    {{
      "question": "Clear, specific question here?",
      "answer": "Direct answer extracted from the text",
      "key_concepts": ["concept1", "concept2"],
      "confidence": 0.95
    }}
  ]
}}

CRITICAL: 
- Answers MUST be grounded in the provided text
- Do NOT make up information
- If you cannot generate {num_questions} high-quality factual questions, generate fewer
- Ensure questions are valuable for evaluating RAG system performance"""

        return self._call_llm_and_parse(prompt, document_id, session_info, "easy", "factual")
    
    def _generate_conceptual_questions(
        self, 
        document_text: str, 
        document_id: str,
        session_info: Dict[str, str],
        num_questions: int
    ) -> List[QuestionAnswer]:
        """Generate conceptual/medium difficulty questions"""
        
        prompt = f"""You are creating evaluation questions for a RAG system based on technical conference content.


TRANSCRIPT SECTION:
{document_text}

TASK: Generate {num_questions} CONCEPTUAL questions that require understanding key ideas and reasoning.

REQUIREMENTS FOR CONCEPTUAL QUESTIONS:
1. Test understanding of WHY/HOW, not just WHAT
2. May require connecting multiple points within this section
3. Focus on design decisions, tradeoffs, best practices
4. Answers should still be grounded in the text but require synthesis
5. Appropriate for senior technical audience

EXAMPLES OF GOOD CONCEPTUAL QUESTIONS:
- "Why did [Speaker] choose approach X over approach Y for this use case?"
- "What are the key tradeoffs when implementing the proposed architecture?"
- "How does the suggested caching strategy improve system performance?"
- "What problem does this design pattern solve?"

FORMAT YOUR RESPONSE AS JSON:
{{
  "questions": [
    {{
      "question": "Thoughtful, conceptual question here?",
      "answer": "Answer that explains the concept/reasoning based on the text",
      "key_concepts": ["concept1", "concept2"],
      "confidence": 0.90
    }}
  ]
}}

CRITICAL:
- Questions should test understanding, not just recall
- Answers must still be derivable from the provided text
- Focus on technical depth appropriate for architects and senior engineers
- Avoid trivial or overly obvious questions"""

        return self._call_llm_and_parse(prompt, document_id, session_info, "medium", "conceptual")
    
    def _generate_complex_questions(
        self, 
        document_text: str, 
        document_id: str,
        session_info: Dict[str, str],
        num_questions: int
    ) -> List[QuestionAnswer]:
        """Generate complex/hard questions"""
        
        prompt = f"""You are creating evaluation questions for a RAG system based on technical conference content.


TRANSCRIPT SECTION:
{document_text}

TASK: Generate {num_questions} COMPLEX questions that require deeper analysis or application.

REQUIREMENTS FOR COMPLEX QUESTIONS:
1. May require synthesizing information from multiple parts of this section
2. Application or scenario-based questions
3. Questions about implementation details or integration
4. Comparison of multiple approaches mentioned
5. Questions that test deep technical understanding

EXAMPLES OF GOOD COMPLEX QUESTIONS:
- "How would you adapt the proposed architecture for a scenario with X constraint?"
- "What are the implementation steps for integrating components A and B discussed in the talk?"
- "How do the performance characteristics of approach X compare to approach Y based on the analysis presented?"
- "What are the prerequisites and considerations before adopting the recommended pattern?"

FORMAT YOUR RESPONSE AS JSON:
{{
  "questions": [
    {{
      "question": "Complex, multi-faceted question here?",
      "answer": "Comprehensive answer that synthesizes information from the text",
      "key_concepts": ["concept1", "concept2", "concept3"],
      "confidence": 0.85
    }}
  ]
}}

CRITICAL:
- These should be the most challenging questions in the test set
- Still must be answerable from the provided text (no external knowledge required)
- Test ability to synthesize and apply information
- Appropriate for technical leaders making architectural decisions"""

        return self._call_llm_and_parse(prompt, document_id, session_info, "hard", "multi_hop")
    
    def _call_llm_and_parse(
        self, 
        prompt: str, 
        document_id: str,
        session_info: Dict[str, str],
        difficulty: str,
        question_type: str
    ) -> List[QuestionAnswer]:
        """Call LLM and parse the response"""
        
        try:

            response_text = self.llm.generate(prompt=prompt, max_tokens=2000)
            
            # Parse JSON response
            # Strip markdown code blocks if present
            #response_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)
            
            # Convert to QuestionAnswer objects
            qa_pairs = []
            for idx, q_data in enumerate(data.get("questions", [])):
                qa = QuestionAnswer(
                    question_id=f"{document_id}_q{idx+1}_{difficulty}",
                    question=q_data["question"],
                    answer=q_data["answer"],
                    source_chunk_id=document_id,
                    source_text="",  # Can be filled later if needed
                    difficulty=difficulty,
                    question_type=question_type,
                    session_info=session_info,
                    confidence_score=q_data.get("confidence", 0.8)
                )
                qa_pairs.append(qa)
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def validate_question(self, qa: QuestionAnswer, document_text: str) -> Dict[str, Any]:
        """
        Validate a generated Q&A pair for quality
        
        Returns dict with validation results
        """
        
        prompt = f"""You are validating a question-answer pair generated for RAG system evaluation.

QUESTION: {qa.question}

ANSWER: {qa.answer}

SOURCE TEXT:
{document_text}

VALIDATION TASK:
Evaluate this Q&A pair on the following criteria:

1. GROUNDEDNESS: Can the answer be fully derived from the source text?
   - Score: 1-5 (1=not grounded, 5=perfectly grounded)
   - Explanation: Brief reason

2. CLARITY: Is the question clear, unambiguous, and well-formed?
   - Score: 1-5 (1=unclear, 5=crystal clear)
   - Explanation: Brief reason

3. ANSWERABILITY: Can this question realistically be answered by a RAG system?
   - Score: 1-5 (1=not answerable, 5=easily answerable)
   - Explanation: Brief reason

4. TECHNICAL_ACCURACY: Is the answer technically correct?
   - Score: 1-5 (1=incorrect, 5=fully correct)
   - Explanation: Brief reason

5. VALUE: Is this question valuable for evaluating RAG performance?
   - Score: 1-5 (1=low value, 5=high value)
   - Explanation: Brief reason

FORMAT YOUR RESPONSE AS JSON:
{{
  "groundedness": {{"score": 5, "explanation": "..."}},
  "clarity": {{"score": 5, "explanation": "..."}},
  "answerability": {{"score": 5, "explanation": "..."}},
  "technical_accuracy": {{"score": 5, "explanation": "..."}},
  "value": {{"score": 5, "explanation": "..."}},
  "overall_score": 4.8,
  "keep": true,
  "issues": ["List any issues found"],
  "suggestions": ["Any suggestions for improvement"]
}}

RECOMMENDATION: Only recommend "keep": true if overall_score >= 4.0"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            #response_text = response_text.replace("```json", "").replace("```", "").strip()
            validation = json.loads(response_text)
            return validation
            
        except Exception as e:
            print(f"Error validating question: {e}")
            return {"keep": False, "overall_score": 0, "issues": [str(e)]}
    
    def generate_and_validate_test_set(
        self,
        chunks: List[Dict[str, Any]],
        validate: bool = True,
        min_score: float = 4.0
    ) -> List[QuestionAnswer]:
        """
        Generate test set from multiple chunks and optionally validate
        
        Args:
            chunks: List of dicts with keys: text, id, session_info
            validate: Whether to run validation
            min_score: Minimum validation score to keep question
        
        Returns:
            List of validated QuestionAnswer objects
        """
        all_questions = []
        
        print(f"Generating questions from {len(chunks)} chunks...")
        
        for idx, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {idx+1}/{len(chunks)}: {chunk.get('id', 'unknown')}")
            
            questions = self.generate_questions_from_document(
                chunk_text=chunk["text"],
                chunk_id=chunk["id"],
                session_info=chunk.get("session_info", {}),
                num_easy=2,
                num_medium=2,
                num_hard=1
            )
            
            print(f"  Generated {len(questions)} questions")
            
            if validate:
                validated_questions = []
                for qa in questions:
                    validation = self.validate_question(qa, chunk["text"])
                    
                    if validation.get("keep") and validation.get("overall_score", 0) >= min_score:
                        validated_questions.append(qa)
                        print(f"    ✓ Kept: {qa.question[:60]}...")
                    else:
                        print(f"    ✗ Filtered: {qa.question[:60]}... (score: {validation.get('overall_score', 0)})")
                
                all_questions.extend(validated_questions)
            else:
                all_questions.extend(questions)
        
        print(f"\n{'='*60}")
        print(f"Total questions generated: {len(all_questions)}")
        print(f"{'='*60}")
        
        return all_questions
    
    def export_test_set(self, questions: List[QuestionAnswer], filename: str):
        """Export test set to JSON file"""
        
        output = {
            "metadata": {
                "total_questions": len(questions),
                "difficulty_distribution": self._get_distribution(questions, "difficulty"),
                "type_distribution": self._get_distribution(questions, "question_type"),
                "generated_by": self.model
            },
            "questions": [asdict(q) for q in questions]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nTest set exported to: {filename}")
        return output
    
    @staticmethod
    def _get_distribution(questions: List[QuestionAnswer], field: str) -> Dict[str, int]:
        """Get distribution of a field across questions"""
        dist = {}
        for q in questions:
            value = getattr(q, field)
            dist[value] = dist.get(value, 0) + 1
        return dist

def load_document_from_file(file: str) -> CrawlPageResult:
    """Load document from file"""
    crawlPageResult = None
    with open(file, "rb") as f:
        data_dict = json.load(f)
        crawlPageResult = CrawlPageResult(**data_dict)
    return crawlPageResult

def generate_questions_and_save_to_file(generator: ConferenceTestSetGenerator, document: CrawlPageResult, 
                                    num_easy: int, num_medium: int, num_hard: int, output_file: str) -> List[QuestionAnswer]:
    """Generate questions from document"""
    start_time = time.perf_counter()
    
    print("---"*30)
    print(f"\nGenerating questions for {document.page_url} Easy: {num_easy}, Medium: {num_medium}, Hard: {num_hard}\n")
    questions = generator.generate_questions_from_document(
            document_text=document.page_content,
            document_id=document.page_url,
            session_info={},
            num_easy=num_easy,
            num_medium=num_medium,
            num_hard=num_hard
    )
    print(f"Generated {len(questions)} questions and saving to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        # Convert each QuestionAnswer object to a dict
        questions_dict = [asdict(q) for q in questions]
        json.dump(questions_dict, f, indent=4)
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("---"*30)
    return questions

def load_documents_from_directory(directory: str) -> List[CrawlPageResult]:
    """Load documents from directory"""
    documents = []
    if os.path.isdir(dir):                    
        md_files = Path(directory).glob("*.md")
        count = 0
        for count, md_file in enumerate(md_files, 1):
            #print(f"index_markdown_file {md_file}\n")
            crawlPageResult = load_document_from_file(md_file)
            documents.append(crawlPageResult)
        print(f"Finished processing {count} files from directory {directory}")
    else:
        print(f"The directory '{directory}' does not exist.")
        raise FileExistsError(f"The directory '{directory}' does not exist.")
    
    return documents
    

def main(file: str, dir: str, num_easy: int, num_medium: int, num_hard: int,  output_dir: str):
    """Main entry point for the script"""    
    print(f"file: {file}, dir: {dir}, num_easy: {num_easy}, num_medium: {num_medium}, num_hard: {num_hard}, output_dir : {output_dir}")

    print("="*60)
    print("RAG Test Set Generator")
    print("="*60)

    # making sure the required files and directories exist
    if not os.path.isfile(file):       
        print(f"The file '{file}' does not exist.")
        raise FileExistsError(f"The file '{file}' does not exist.")
    
    if not os.path.isdir(dir):       
        print(f"The directory '{dir}' containing the md files does not exist.")
        raise FileExistsError(f"The directory '{dir}' containing the md files does not exist.")

    if not os.path.isdir(output_dir):
        print(f"The directory '{output_dir}' to save the test questions does not exist.")
        raise FileExistsError(f"The directory '{output_dir}' to save the test questions does not exist.")

    
    md_file_list = []
    try:
        with open(file, 'r') as f:
            md_file_list = json.load(f)        
    except json.JSONDecodeError:
        print(f"Error: The file '{file}' is not a valid JSON file.")    

    print(f"Found a total of {len(md_file_list)} md files to process")

    # Initialize LLM
    print(f"\nInitializing LLM")
    gemini_llm_provider_info = get_llm_provider("gemini")
    
    gemini_llm = RagLLM.create(
        base_url=gemini_llm_provider_info.base_url,
        api_key=gemini_llm_provider_info.api_key,
        model_name=gemini_llm_provider_info.model_name
    )

    fireworks_llm_provider_info = get_llm_provider("fireworks")
    fireworks_llm = RagLLM.create(
        base_url=fireworks_llm_provider_info.base_url,
        api_key=fireworks_llm_provider_info.api_key,
        model_name=fireworks_llm_provider_info.model_name
    )

    # Initialize generator
    print(f"\nInitializing generator")
    gemini_generator = ConferenceTestSetGenerator(llm=gemini_llm)
    fireworks_generator = ConferenceTestSetGenerator(llm=fireworks_llm)

    for count, md_file in enumerate(md_file_list, 1):
        md_file_path = f"{dir}/{md_file}"
        print(f"\nProcessing file: {md_file_path}")
        crawlPageResult = load_document_from_file(md_file_path)
        output_file = os.path.basename(md_file).replace(".md", "_questions.json")
       
        generate_questions_and_save_to_file(fireworks_generator, crawlPageResult, num_easy, num_medium, num_hard, f"{output_dir}/{output_file}")
        print(f"✓ Generated questions for {md_file}")
    
    print(f"Processed a total of {len(md_file_list)} md files")
    print("====== done processing =============")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test questions for evaluation"
    )

    parser.add_argument("--file", "-f", help="File contains a list of md file names to generate test questions", required=True)
    parser.add_argument( "--dir", "-d", help="Directory containing md files to generate test questions", required=False)
    parser.add_argument("--num_easy", "-e", help="Number of easy questions to generate", required=False, default=2)
    parser.add_argument("--num_medium", "-m", help="Number of medium questions to generate", required=False, default=2)
    parser.add_argument("--num_complex", "-c", help="Number of hard questions to generate", required=False, default=0)
    parser.add_argument("--output", "-o", help="A directory to save the test questions to", required=False)
    args = parser.parse_args()
    print(args)

    if not args.file and not args.dir and not args.output:
        print("Please provide  --file and --dir and --output argument to generate test questions")
        raise ValueError("Please provide either --file or --dir argument to generate test questions")
    
    main(args.file, args.dir, args.num_easy, args.num_medium, args.num_complex, args.output)
    