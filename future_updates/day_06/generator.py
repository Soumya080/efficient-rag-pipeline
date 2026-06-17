"""
Generator for RAG pipeline.

Builds RAG prompts and calls LLM for answer generation.
"""


def build_rag_prompt(query, context):
    """
    Build a RAG prompt with context and query.

    User's original: build_rag_prompt()

    Args:
        query: User's question
        context: Retrieved context string

    Returns:
        Formatted prompt string
    """
    prompt = f"""
You are a retrieval-augmented AI assistant.

You MUST answer ONLY from the provided context.

RULES:
1. Use ONLY information from the context
2. Do NOT use prior knowledge
3. Do NOT infer beyond the context
4. If answer is missing, say:
   "I could not find the answer in the provided context."

==================== CONTEXT ====================

{context}

==================== QUESTION ====================

{query}

==================== INSTRUCTIONS ====================

Return a concise answer strictly grounded in the context.

==================== ANSWER ====================
"""
    return prompt


def generate_answer(prompt, model_name="phi3"):
    """
    Call Ollama LLM to generate an answer.

    User's original code:
        response = ollama.chat(
            model='phi3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        print(response['message']['content'])

    Args:
        prompt: The full RAG prompt string
        model_name: Ollama model name (default: phi3)

    Returns:
        Generated answer string
    """
    try:
        import ollama

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    except ImportError:
        print("[WARNING] ollama package not installed.")
        print("Install with: pip install ollama")
        print("Make sure Ollama is running locally.")
        return f"[OLLAMA NOT AVAILABLE]"

    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"Make sure model is pulled: ollama pull {model_name}")
        return f"[ERROR] {e}"
