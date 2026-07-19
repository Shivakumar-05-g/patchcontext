import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src import config
from src.retriever import retrieve_context

logger = logging.getLogger(__name__)

def get_llm():
    """Initialize the ChatGroq model using configured parameters."""
    if not config.GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable is not set!")
        raise ValueError("GROQ_API_KEY not found in environment. Please add it to your .env file.")
        
    logger.info(f"Initializing ChatGroq LLM (model: {config.GROQ_MODEL})...")
    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.GROQ_MODEL,
        temperature=0.0  # Zero temperature for deterministic grounding
    )

def format_context_for_prompt(docs):
    """Formats retrieved Document objects into a clean text block for the LLM context."""
    formatted_chunks = []
    for idx, doc in enumerate(docs):
        source_type = doc.metadata.get("source_type", "unknown")
        source_url = doc.metadata.get("source_url", "n/a")
        
        # Build document header
        header = f"--- Retrieved Document [{idx+1}] | Type: {source_type} | URL: {source_url} ---"
        
        # Append specific identifier metadata
        meta_details = []
        if source_type == "commit":
            meta_details.append(f"Commit SHA: {doc.metadata.get('commit_sha', 'n/a')}")
        elif source_type == "pull_request":
            meta_details.append(f"PR Number: #{doc.metadata.get('pr_number', 'n/a')}")
        elif source_type == "issue":
            meta_details.append(f"Issue Number: #{doc.metadata.get('issue_number', 'n/a')}")
            
        meta_str = ", ".join(meta_details)
        chunk_text = f"{header}\nMetadata: {meta_str}\nContent:\n{doc.page_content}\n"
        formatted_chunks.append(chunk_text)
        
    return "\n".join(formatted_chunks)

def generate_answer(question, k=config.DEFAULT_K, fetch_k=config.DEFAULT_FETCH_K):
    """Retrieve context and generate grounded answer using ChatGroq."""
    logger.info(f"Generating answer for: '{question}'...")
    
    # 1. Retrieve relevant documents using MMR
    retrieved_docs = retrieve_context(question, k=k, fetch_k=fetch_k)
    
    if not retrieved_docs:
        return {
            "question": question,
            "answer": "I could not retrieve any relevant repository history documents to answer this question.",
            "retrieved_docs": []
        }
        
    # 2. Format the context
    context_text = format_context_for_prompt(retrieved_docs)
    
    # 3. Create the chat prompt
    # Define robust grounding instructions
    system_prompt = (
        "You are PatchContext, a historical developer assistant for the FastAPI repository.\n"
        "Your task is to answer the user's question about design decisions, behavior changes, or code implementations "
        "using ONLY the retrieved FastAPI repository history context provided below.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        "{context}\n"
        "=========================\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Ground your answer strictly in the provided retrieved context. Do not make up information.\n"
        "2. Do not invent commit SHAs, PR numbers, issue numbers, URLs, authors, or discussions. Only cite what is explicitly present in the metadata or content above.\n"
        "3. If the retrieved context is insufficient to answer the question, state clearly: 'Based on the retrieved context, I do not have enough information to answer this question.' Do not attempt to answer using general knowledge.\n"
        "4. Distinguish clearly between facts directly supported by the context and your interpretations or deductions. Be objective.\n"
        "5. You must cite the sources of your claims inline. Format your citations exactly as:\n"
        "   - [PR: #Number] (e.g. [PR: #16021]) for Pull Request facts.\n"
        "   - [Issue: #Number] (e.g. [Issue: #15974]) for Issue facts.\n"
        "   - [Commit: SHA] (e.g. [Commit: 7fe315c21afb8a57a2b559772e0f7ced7e5d071a]) for Commit facts.\n"
        "6. Do not include citations that are not present in the retrieved context metadata."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}")
    ])
    
    # 4. Instantiate LLM and chain
    try:
        llm = get_llm()
        chain = prompt_template | llm
        
        # Invoke chain
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        
        answer_text = response.content
        logger.info("Answer generated successfully.")
        
        return {
            "question": question,
            "answer": answer_text,
            "retrieved_docs": retrieved_docs
        }
        
    except Exception as e:
        logger.error(f"Error in RAG generation: {e}")
        return {
            "question": question,
            "answer": f"An error occurred while generating the answer: {e}",
            "retrieved_docs": retrieved_docs
        }
