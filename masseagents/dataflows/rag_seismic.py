# -*- coding: utf-8 -*-
import os
import json
import faiss
import fitz
import numpy as np
import tiktoken
from typing import Dict, Any
from openai import OpenAI
from anthropic import Anthropic

class RAGSeismicInterface:
    """RAG-based seismic parameter extraction from BCBC PDF"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize appropriate client based on model
        self.chat_model = config.get("llm_model", "gpt-4o")
        if self.chat_model.startswith("claude-"):
            # Use Anthropic client for Claude models
            self.client = Anthropic(api_key=config.get("llm_providers", {}).get("anthropic", {}).get("api_key"))
            self.openai_client = OpenAI(api_key=config.get("llm_providers", {}).get("openai", {}).get("api_key"))
        else:
            # Use OpenAI client for other models
            self.client = OpenAI(api_key=config.get("llm_providers", {}).get("openai", {}).get("api_key"))
            self.openai_client = self.client
        
        # Configuration parameters
        self.pdf_path = os.path.join(os.path.dirname(__file__), "bcbc_2018dbac.pdf")
        self.vector_db_dir = os.path.join(os.path.dirname(__file__), "vector_db")
        # Embedding model should always be fixed for consistency
        self.embedding_model = "text-embedding-3-small"
        # Use parameters from config to ensure consistency across all models
        self.max_tokens = config.get("max_tokens", 2000)  # Increased default
        self.chunk_overlap = 100
        self.top_k = config.get("top_k", 6)
        # Handle model-specific temperature settings
        if self.chat_model == "o4-mini" or self.chat_model == "gpt-5":
            self.temperature = 1  # o4-mini and gpt-5 only support temperature=1
        else:
            self.temperature = 0
        # Increase max_response_tokens to prevent empty responses - unified for all models
        if self.chat_model == "o4-mini" or self.chat_model == "gpt-5" or self.chat_model.startswith("claude-"):
            self.max_response_tokens = 2000  # Much larger for o4-mini, gpt-5, and claude to ensure complete JSON response
        else:
            self.max_response_tokens = 800  # Increased for other models
        
        # Load or build vector database
        self.index, self.text_chunks = self._load_or_build_vector_db()

    def _load_or_build_vector_db(self):
        """Load existing vector DB or build from PDF"""
        index_path = os.path.join(self.vector_db_dir, "faiss_index.bin")
        chunks_path = os.path.join(self.vector_db_dir, "text_chunks.json")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"📚 Loading existing vector database from {self.vector_db_dir}")
            index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            return index, chunks

        print(f"🔨 Building new vector database from {self.pdf_path}")
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
        text = self._extract_text_from_pdf(self.pdf_path)
        chunks = self._split_text_into_chunks(text)
        embeddings = [self._get_embedding(chunk) for chunk in chunks]
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        os.makedirs(self.vector_db_dir, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"✅ Vector database built and saved to {self.vector_db_dir}")
        return index, chunks

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        print(f"📖 Extracting text from PDF: {pdf_path}")
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        print(f"📄 Extracted {len(text)} characters from PDF")
        return text

    def _split_text_into_chunks(self, text: str) -> list:
        """Split text into chunks with overlap"""
        print("✂️ Splitting text into chunks...")
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk = encoder.decode(tokens[start:end])
            chunks.append(chunk)
            start += self.max_tokens - self.chunk_overlap
        print(f"📋 Created {len(chunks)} text chunks")
        return chunks

    def _get_embedding(self, text: str) -> list:
        """Get embedding vector for text"""
        response = self.openai_client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def extract_seismic_parameters(self, location: str) -> Dict[str, Any]:
        """RAG extraction of seismic parameters based on city name"""
        print(f"🔍 Searching for seismic parameters for: {location}")
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(location + " seismic parameters")
            
            # Search for relevant chunks
            D, I = self.index.search(np.array([query_embedding]).astype("float32"), k=self.top_k)
            relevant_chunks = [self.text_chunks[i] for i in I[0]]
            
            # Build context from relevant chunks
            context = "\n\n".join([f"Document Segment {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
            
            print(f"📚 Found {len(relevant_chunks)} relevant document segments")
            
            # Create prompt for parameter extraction
            user_message = f"""Extract seismic parameters for {location} from the document below.

Document:
{context}

Find the exact numerical values for {location} and return ONLY this JSON:

{{
  "Sa_02": <number>,
  "Sa_05": <number>, 
  "Sa_10": <number>,
  "Sa_20": <number>,
  "PGA": <number>,
  "PGV": <number>
}}

Rules:
- Return ONLY the JSON object
- Use actual numerical values from the document
- If {location} is not found, return: {{"error": "City not found"}}
- No explanations, no text outside JSON"""

            # Handle model-specific parameters
            if self.chat_model.startswith("claude-"):
                # Claude models use Anthropic client
                response = self.client.messages.create(
                    model=self.chat_model,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                    system="You extract seismic data from building codes. Return only JSON. No explanations.",
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                content = response.content[0].text.strip()
            elif self.chat_model == "o4-mini" or self.chat_model == "gpt-5":
                # o4-mini and gpt-5 use max_completion_tokens instead of max_tokens
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": "You extract seismic data from building codes. Return only JSON. No explanations."},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_response_tokens
                )
                content = response.choices[0].message.content.strip()
            else:
                # Other models use max_tokens
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": "You extract seismic data from building codes. Return only JSON. No explanations."},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_response_tokens
                )
                content = response.choices[0].message.content.strip()
            
            # Check if response is empty
            if not content:
                raise ValueError("Empty response from API")
            
            # Clean response content
            json_str = content.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            if not json_str:
                raise ValueError("Empty response from API")
                
            result = json.loads(json_str)
            
            if "error" in result:
                print(f"❌ {result['error']}")
                raise ValueError(result["error"])
            else:
                print(f"✅ Successfully extracted seismic parameters for {location}")
                return result
                
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"RAG extraction error: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)

    def get_available_locations(self) -> list:
        """Get list of available locations in the database"""
        # This could be enhanced to actually scan the PDF for city names
        # For now, return known Canadian cities from BCBC
        return [
            "Vancouver", "Nanaimo", "Victoria", "Toronto", "Montreal", 
            "Calgary", "Edmonton", "Winnipeg", "Halifax", "St. John's"
        ]

    def validate_database(self) -> bool:
        """Validate that the vector database is properly loaded"""
        try:
            if self.index is None or self.text_chunks is None:
                return False
            if self.index.ntotal == 0 or len(self.text_chunks) == 0:
                return False
            return True
        except Exception:
            return False 