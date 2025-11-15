# eduplanner/agents/skilltree.py
import json
from llm_connector import LLMConnector

class SkillTreeAgent:
    """Generates the initial student skill tree using DeepSeek 70B."""

    def __init__(self, subject_schema_path: str):
        self.skill_nodes = self._load_skill_nodes(subject_schema_path)

    def _load_skill_nodes(self, path: str) -> list:
        """Loads the predefined skill node names from a subject JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get("skill_nodes", [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Subject schema file not found at: {path}")

    def generate(self, student_level: str) -> dict:
        """Generates and returns the initial skill tree."""
        node_list = ", ".join(self.skill_nodes)
        
        prompt = f"""
        You are an OS tutor. The student is {student_level}.
        Build a Skill-Tree with the following 5 nodes: {node_list}.
        Rate the student's expected capability for each node on a scale of 0 (no knowledge) to 5 (expert).
        Output the result ONLY as a JSON object where keys are the skill names and values are the scores.
        """
        
        # Use the expert model (DeepSeek) for this creation task
        skill_tree_json = LLMConnector.get_expert_response(prompt, is_json=True)
        
        if skill_tree_json is None:
            # Fallback for API failure: return a default, low-score skill tree
            print("Warning: SkillTree generation failed. Returning default low scores.")
            return {node: 1 for node in self.skill_nodes}

        return skill_tree_json

    @staticmethod
    def update_tree(current_tree: dict, quiz_accuracy: float) -> dict:
        """Simulates updating the skill tree based on quiz performance."""
        # Simple reinforcement: increase score by 1 for nodes > 0 if accuracy > 80%
        # A more complex system would map questions to specific nodes.
        new_tree = current_tree.copy()
        if quiz_accuracy >= 0.8:
            for skill, score in new_tree.items():
                if score < 5:
                    # Increment score by 1, or 2 if performance was excellent
                    increment = 1 if quiz_accuracy < 1.0 else 2 
                    new_tree[skill] = min(5, score + increment)
        
        return new_tree