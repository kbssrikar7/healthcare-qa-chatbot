"""
LangChain prompt templates for Medical QA.

Provides ChatPromptTemplate versions of the existing prompts
with proper message roles for chat models.
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


# Medical QA System Prompt
MEDICAL_SYSTEM_PROMPT = """You are a knowledgeable medical assistant. Your role is to provide accurate, helpful health information based on the context provided. Always be clear about limitations and recommend consulting healthcare professionals for medical decisions.

### Important Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Use clear, patient-friendly language
4. Include relevant medical terms with explanations
5. NEVER provide diagnoses or prescriptions
6. Always recommend consulting a healthcare professional for specific medical advice"""


# Medical QA Chat Template
MEDICAL_QA_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
    HumanMessage(content="""### Context:
{context}

### Question:
{question}

Please provide a helpful, accurate answer based on the context above.""")
])


# Explainable QA Chat Template (requests citations and confidence)
EXPLAINABLE_QA_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a knowledgeable medical assistant focused on providing transparent, explainable answers.

### Important Guidelines:
1. Answer the question based on the provided context
2. Cite which parts of the context support your answer
3. Indicate your confidence level (High/Medium/Low)
4. Note any limitations or uncertainties
5. Use clear, patient-friendly language
6. NEVER provide diagnoses or prescriptions"""),
    HumanMessage(content="""### Context:
{context}

### Question:
{question}

Please provide an answer with citations and confidence level.""")
])


# Simple QA Template (for fast responses)
SIMPLE_QA_TEMPLATE = PromptTemplate.from_template(
    """Context: {context}

Question: {question}

Provide a helpful, accurate answer based on the context above. Be concise but thorough.

Answer:"""
)


# Grounding Gate Prompt (checks if context is sufficient)
GROUNDING_CHECK_TEMPLATE = PromptTemplate.from_template(
    """Based on the following context, determine if there is sufficient information to answer the question.

Context:
{context}

Question:
{question}

Can this question be answered from the context? Respond with only 'YES' or 'NO'."""
)


# Rationale Generation Template
RATIONALE_TEMPLATE = PromptTemplate.from_template(
    """Based on the following question, answer, and context, explain the reasoning process.

Question: {question}

Answer: {answer}

Context Used:
{context}

Provide a brief explanation of:
1. What key information from the context supports this answer
2. Any assumptions or limitations in the answer
3. Why this answer is appropriate given the available information

Reasoning:"""
)


# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here.
"""


def get_prompt_template(template_name: str = "medical_qa") -> ChatPromptTemplate:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template (medical_qa, explainable, simple)
        
    Returns:
        The requested prompt template
    """
    templates = {
        "medical_qa": MEDICAL_QA_CHAT_TEMPLATE,
        "explainable": EXPLAINABLE_QA_CHAT_TEMPLATE,
        "simple": SIMPLE_QA_TEMPLATE
    }
    return templates.get(template_name, MEDICAL_QA_CHAT_TEMPLATE)


def format_context_for_prompt(documents: list, max_length: int = 2000) -> str:
    """
    Format documents into a context string for prompts.
    
    Args:
        documents: List of LangChain Documents
        max_length: Maximum context length
        
    Returns:
        Formatted context string
    """
    context_parts = []
    total_length = 0
    
    for i, doc in enumerate(documents):
        # Handle both LangChain Documents and dicts
        if hasattr(doc, 'page_content'):
            content = doc.page_content
            source = doc.metadata.get("source", f"Source {i+1}")
        elif isinstance(doc, dict):
            content = doc.get("content", str(doc))
            source = doc.get("source", f"Source {i+1}")
        else:
            content = str(doc)
            source = f"Source {i+1}"
        
        entry = f"[{source}]: {content}"
        
        if total_length + len(entry) > max_length:
            break
        
        context_parts.append(entry)
        total_length += len(entry)
    
    return "\n\n".join(context_parts)
