# eduplanner/agents/evaluator.py
from llm_connector import LLMConnector

class EvaluatorAgent:
    """Evaluates the lesson plan based on 5D (CIDDP) metrics."""

    EVAL_PROMPT_TEMPLATE = """
    Evaluate the following lesson plan using the 5 CIDDP metrics:
    - Clarity: How easy is the structure and language to follow?
    - Integrity: Is the content factually correct and consistent?
    - Depth: Is the complexity appropriate for the target level (consider skill tree)?
    - Practicality: Does it include real-world examples or exercises?
    - Pertinence: Are the topics relevant and logically sequenced?
    
    Target Skill Tree: {skill_tree}
    Lesson Plan to Evaluate:
    ---
    {lesson_plan}
    ---

    Return scores (0-10 each) and detailed feedback.
    Output ONLY a JSON object with keys: "Clarity", "Integrity", "Depth", "Practicality", "Pertinence", "Average", "Advantages" (list), "Disadvantages" (list).
    The "Average" key should be the mean of the five scores.
    """

    @staticmethod
    def evaluate(lesson_plan: str, skill_tree: dict) -> dict:
        """Sends the plan and skill tree to the agent model for evaluation."""
        prompt = EvaluatorAgent.EVAL_PROMPT_TEMPLATE.format(
            lesson_plan=lesson_plan,
            skill_tree=skill_tree
        )
        # Use the cost-effective agent model (OpenAI)
        return LLMConnector.get_agent_response(prompt, is_json=True)