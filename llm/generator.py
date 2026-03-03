import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

class CurriculumLLM:
    """
    Generates exam-oriented, step-by-step answers using Groq LLM,
    strictly based on the provided curriculum context.
    """

    def __init__(self, model_id="llama-3.3-70b-versatile"):
        """
        Initializes the Groq client.
        
        Args:
            model_id: The Groq model to use. Options include:
                - "llama-3.3-70b-versatile" (recommended, strong reasoning)
                - "llama-3.1-8b-instant" (faster, lighter)
                - "mixtral-8x7b-32768" (good balance)
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Get your free API key from https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        print(f"Groq LLM initialized with model '{model_id}'")

    def generate_answer(self, query, context_chunks):
        """
        Generates a structured, exam-oriented answer using Groq LLM
        and retrieved curriculum context.

        Args:
            query: The user's question.
            context_chunks: List of dicts with 'text' key from the retriever.

        Returns:
            A step-by-step answer string, or a refusal message if out of scope.
        """

        # If no relevant context was retrieved, refuse to answer
        if not context_chunks:
            return (
                "This question is outside the scope of the curriculum. "
                "I can only answer questions related to the syllabus."
            )

        # Combine retrieved curriculum text
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

        # System prompt: strict curriculum-native behavior
        system_prompt = (
            "You are an expert tutor that answers questions STRICTLY based on the "
            "provided curriculum context. Follow these rules:\n"
            "1. ONLY use information from the provided context. Do NOT add outside knowledge.\n"
            "2. Provide step-by-step, exam-oriented answers.\n"
            "3. Use formulas, definitions, and examples from the context when available.\n"
            "4. If the context does not contain enough information to answer, say so clearly.\n"
            "5. Keep answers clear, structured, and suitable for exam preparation.\n"
            "6. Do NOT hallucinate or make up information.\n"
        )

        # User prompt with context and question
        user_prompt = (
            f"CURRICULUM CONTEXT:\n"
            f"---\n{context_text}\n---\n\n"
            f"STUDENT QUESTION: {query}\n\n"
            f"Provide a clear, step-by-step answer based ONLY on the above context."
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_id,
                temperature=0.2,   # Low temperature for factual accuracy
                max_tokens=1024,
                top_p=1,
            )
            answer = chat_completion.choices[0].message.content.strip()
            return answer

        except Exception as e:
            print(f"Groq API error: {e}")
            return f"Error generating answer: {str(e)}"


# Quick test
if __name__ == "__main__":
    llm = CurriculumLLM()
    sample_context = [
        {"text": "Kinematics is the branch of classical mechanics that describes the motion of points, bodies, and systems of bodies without considering the forces that cause them to move."}
    ]
    answer = llm.generate_answer("What is kinematics?", sample_context)
    print("Answer:", answer)
