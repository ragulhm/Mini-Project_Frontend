# eduplanner/agents/analyst.py
from llm_connector import LLMConnector

class AnalystAgent:
    """Detects and lists common student mistakes/misconceptions for the topics."""

    ANALYST_PROMPT_TEMPLATE = """
    Analyze the following lesson plan to identify 2-3 common beginner mistakes or misconceptions 
    for the primary concepts mentioned in the plan (e.g., Process Management, Memory Management).
    
    Format the output ONLY as a JSON object where keys are the concepts (e.g., "Paging") 
    and the values are a list of the common misconceptions.
    
    Lesson Plan:
    ---
    {lesson_plan}
    ---
    """

    @staticmethod
    def analyze(lesson_plan: str) -> dict:
        """Generates the list of common errors."""
        prompt = AnalystAgent.ANALYST_PROMPT_TEMPLATE.format(lesson_plan=lesson_plan)
        
        # Use the cost-effective agent model (OpenAI)
        return LLMConnector.get_agent_response(prompt, is_json=True)

    @staticmethod
    def format_errors(error_dict: dict) -> str:
        """Formats the error dictionary into a clean string to append to the plan."""
        if not error_dict:
            return ""
            
        output = "\n\n=== 🧠 AVOID THESE COMMON MISTAKES ===\n"
        for topic, errors in error_dict.items():
            output += f"\n👉 **{topic}**:\n"
            for i, error in enumerate(errors):
                output += f"   - {error}\n"
        return output