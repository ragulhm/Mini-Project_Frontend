# eduplanner/agents/optimizer.py
from llm_connector import LLMConnector

class OptimizerAgent:
    """Improves the lesson plan based on Evaluator Agent feedback."""

    OPTIMIZE_PROMPT_TEMPLATE = """
    You are an Optimizer Agent for an educational planner. Your goal is to improve a lesson plan.
    
    Use the provided feedback to generate an improved version of the original plan.
    Crucially, only output the full, revised lesson plan text, without any introductory or concluding remarks.
    
    Evaluator Feedback:
    ---
    {feedback}
    ---
    
    Original Lesson Plan:
    ---
    {original_plan}
    ---
    
    Improved Lesson Plan:
    """

    @staticmethod
    def optimize(original_plan: str, feedback: dict) -> str:
        """Generates the improved lesson plan."""
        # Format the feedback into a readable string
        feedback_str = f"Average Score: {feedback.get('Average', 'N/A')}\n"
        feedback_str += "Disadvantages to Fix:\n"
        for item in feedback.get('Disadvantages', []):
            feedback_str += f"- {item}\n"

        prompt = OptimizerAgent.OPTIMIZE_PROMPT_TEMPLATE.format(
            original_plan=original_plan,
            feedback=feedback_str
        )
        
        # Use the cost-effective agent model (OpenAI)
        return LLMConnector.get_agent_response(prompt)