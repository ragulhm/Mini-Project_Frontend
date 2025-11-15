import os
import requests
import json
import random
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- Load Environment Variables from .env file ---
load_dotenv()

# ==============================================================================
# I. Configuration & Utilities (OpenRouter API Setup)
# ==============================================================================

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Check your .env file.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model Mapping based on Paper
MODEL_MAP = {
    "Evaluator": {"model": "meta-llama/llama-3.3-70b-instruct:free", "temp": 0.0},
    "Optimizer": {"model": "openai/gpt-oss-20b:free", "temp": 1.0},
    "Analyst": {"model": "openai/gpt-oss-20b:free", "temp": 0.7}
}

def call_llm(system_prompt: str, user_prompt: str, agent_type: str) -> str:
    """Generic function to call the LLM via OpenRouter."""
    config = MODEL_MAP.get(agent_type)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config["temp"],
        "max_tokens": 1500 
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status() 
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        # Logging to the console for debugging
        print(f"LLM API Error for {agent_type}: {e}") 
        return f"LLM_ERROR: Failed to generate response ({e})."


# ==============================================================================
# II. Data & Structure Utilities (Including CIDDP Parsing)
# ==============================================================================

def load_questions(file_path: str):
    """Loads the question data from a JSON file."""
    try:
        # NOTE: This runs when the server starts.
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Question file not found at {file_path}.")
        return []

def select_analyst_question(questions: list, topic: str) -> dict:
    """Selects a question matching the current lesson topic for the Analyst Agent."""
    topic_questions = [q for q in questions if q['topic'] == topic]
    if topic_questions:
        return random.choice(topic_questions)
    return random.choice(questions) if questions else None

CIDDP_STANDARD = """
Evaluation Standard (Points 1-100):
1. Clarity (C): Directness and simplicity, avoiding complexity and redundancy.
2. Integrity (I): Completeness and systematic coverage of both knowledge points and exercises.
3. Depth (D): Ability to inspire deep thinking and facilitate understanding of underlying connections.
4. Practicality (P): Practical application value of examples to solve real-life problems.
5. Pertinence (Pe): Adaptability to the student's Skill-Tree levels and learning needs.
Workflow: Output your final verdict in the following exact format:
"[C]: [points]; [short analyzes]”
"[I]: [points]; [short analyzes]”
"[D]: [points]; [short analyzes]”
"[P]: [points]; [short analyzes]”
"[Pe]: [points]; [short analyzes]"
Followed by a summary of advantages and disadvantages.
"""

def parse_ciddp_output(text: str) -> (int, str, dict):
    """Parses the structured CIDPP output from the Evaluator Agent."""
    scores = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        try:
            key_part, rest = line.split(":", 1)
            key = key_part.strip().strip("[]")
            points_part, _ = rest.split(";", 1)
            scores[key] = int(points_part.strip())
        except:
            continue 

    avg_score = sum(scores.values()) / len(scores) if len(scores) > 0 else 0
    feedback_start = text.find("Advantage:")
    feedback_f = text[feedback_start:] if feedback_start != -1 else "No summary feedback provided."

    return round(avg_score), feedback_f, scores

# ==============================================================================
# III. Skill-Tree Structure (OS Domain)
# ==============================================================================

class SkillTree:
    def __init__(self, topic: str, scores: dict):
        self.topic = topic
        self.scores = {
            "Processes & Threads": scores.get("Processes & Threads", 4),      
            "Memory Management": scores.get("Memory Management", 3),          
            "Concurrency & Sync": scores.get("Concurrency & Sync", 2),        
            "File System & I/O": scores.get("File System & I/O", 4),          
            "OS Fundamentals": scores.get("OS Fundamentals", 5)                
        }
        self.level = list(scores.values())

    def to_prompt_string(self) -> str:
        """Generates a structured string to inject into LLM prompts."""
        details = "\n".join([f"- {k}: Level {v}" for k, v in self.scores.items()])
        return f"""
        Student Profile (Skill-Tree Analysis - Operating Systems):
        Teaching Topic: {self.topic}
        Core OS Abilities (Scores 1-5, where 5 is highest):
        {details}
        """

# ==============================================================================
# IV. Multiagent System Classes (Evaluator, Analyst, Optimizer)
# ==============================================================================
# (Classes remain the same)

class EvaluatorAgent:
    def __init__(self, skill_tree: SkillTree):
        self.agent_type = "Evaluator"
        self.skill_tree_prompt = skill_tree.to_prompt_string()

    def evaluate(self, lesson_plan: str) -> (int, str, dict):
        system_prompt = f"""Role: You are an impartial evaluator, experienced in educational content analysis and instructional design evaluation. You will assess a Lesson Plan based on the student's Skill-Tree and the 5-D CIDDP standard.
        {self.skill_tree_prompt}{CIDDP_STANDARD}Also, provide a summary of the overall advantages ('Advantage:') and disadvantages ('Disadvantage:') of the lesson plan."""
        user_prompt = f"Lesson Plan to Evaluate:\n---\n{lesson_plan}\n---\nPerform the 5-D evaluation (C, I, D, P, Pe). Then, list key overall advantages and disadvantages of this plan for the student. Output MUST start with the 5-D scores in the exact format, followed by Advantages and Disadvantages."
        full_response = call_llm(system_prompt, user_prompt, self.agent_type)
        return parse_ciddp_output(full_response)

class AnalystAgent:
    def __init__(self, skill_tree: SkillTree):
        self.agent_type = "Analyst"
        self.skill_tree_prompt = skill_tree.to_prompt_string()

    def analyze_errors(self, question_text: str) -> str:
        system_prompt = f"""Role: You are a Question Analyst agent. Your task is to analyze the provided Operating Systems question and predict the top 3 common error-prone points (mistakes) this student would likely make.
        The student's background is: {self.skill_tree_prompt}
        Output the mistakes in order of probability from largest to smallest.
        Format each mistake as: 'Common Mistake [Mistake #]: [Detailed Description of Mistake] ([Estimated Probability %])'."""
        user_prompt = f"Analyze the following example question and predict the top 3 common student mistakes:\nQuestion: {question_text}"
        return call_llm(system_prompt, user_prompt, self.agent_type)

class OptimizerAgent:
    def __init__(self, skill_tree: SkillTree):
        self.agent_type = "Optimizer"
        self.skill_tree_prompt = skill_tree.to_prompt_string()
        self.queue = [] 
        self.queue_capacity = 5 

    def _update_queue(self, new_lp: str, new_score: int):
        self.queue.append((new_lp, new_score))
        self.queue.sort(key=lambda x: x[1], reverse=True) 
        self.queue = self.queue[:self.queue_capacity]

    def generate_initial_lp(self, topic: str, initial_prompt: str) -> str:
        system_prompt = f"""Role: You are a professional instructional design expert for Operating Systems. Your task is to create a high-quality, concise Lesson Plan for the topic '{topic}', tailored to the student: {self.skill_tree_prompt}.
        The Lesson Plan must contain EXACTLY two parts: 1. Part1: Explanation of knowledge points. 2. Part2: Exercise explanation. Limited to about 300 words."""
        user_prompt = f"Generate the initial instructional design based on the topic and student profile. Focus: {initial_prompt}"
        return call_llm(system_prompt, user_prompt, self.agent_type)

    def optimize_lp(self, previous_lp: str, feedback_f: str, analyst_insert: str = "") -> str:
        system_prompt = f"""Role: You are a Lesson Plan Optimizer agent. Your goal is to maximize the evaluation score. The plan must be tailored to the student: {self.skill_tree_prompt}.
        Optimization task: Improve the previous Lesson Plan based on the feedback. {analyst_insert}"""
        user_prompt = f"Previous Lesson Plan:\n---\n{previous_lp}\n---\nEvaluator Feedback (F):\n---\n{feedback_f}\n---\nGenerate a NEW, optimized Lesson Plan (lp_new). It MUST address the specific disadvantages and integrate the common student mistakes from the Analyst Insertion."
        return call_llm(system_prompt, user_prompt, self.agent_type)

# ==============================================================================
# V. Core Optimization Loop (Refactored to be a function for Flask)
# ==============================================================================

def run_optimization_core(student_skills: SkillTree, lesson_topic: str, questions_data: list, max_iterations: int = 4):
    """Refactored loop that performs the multiagent optimization and returns the result."""
    
    # Initialize Agents
    evaluator = EvaluatorAgent(student_skills)
    optimizer = OptimizerAgent(student_skills)
    analyst = AnalystAgent(student_skills)
    
    # --- Generate Initial LP (Lp0) ---
    initial_prompt = f"Focus on core concepts and examples for {lesson_topic}."
    current_lp = optimizer.generate_initial_lp(lesson_topic, initial_prompt)
    
    best_lp = current_lp
    best_score = 0
    
    for i in range(max_iterations):
        # 1. Evaluate
        score, feedback_f, structured_output = evaluator.evaluate(current_lp)
        optimizer._update_queue(current_lp, score)
        if score > best_score:
            best_score = score
            best_lp = current_lp
        
        # 2. Analyze
        analyst_q_data = select_analyst_question(questions_data, lesson_topic)
        analyst_q_text = analyst_q_data['question'] if analyst_q_data else "Placeholder question for OS concept."
        error_analysis = analyst.analyze_errors(analyst_q_text)
        
        # 3. Prepare Analyst Insertion
        analyst_insert = f"""
        --- ANALYST AGENT INSERTION ---
        The core example question to be explained in Part2 is: '{analyst_q_text}'
        Common Student Mistakes for this question (to be integrated into explanation):
        {error_analysis}
        -----------------------------------
        """
        
        # 4. Optimize
        current_lp = optimizer.optimize_lp(current_lp, feedback_f, analyst_insert)
        
    return best_lp, best_score, student_skills.to_prompt_string()

# ==============================================================================
# VI. Flask Server Endpoints
# ==============================================================================

app = Flask(__name__)
CORS(app) 

# Global variable to store question data loaded at startup
os_questions = load_questions("os_questions.json")


# NEW ROUTE: Serves the index.html file (frontend UI)
@app.route('/')
def serve_index():
    # Flask looks in the 'templates' folder for this file
    return render_template('index.html')

# NEW ROUTE: Serves the os_questions.json file (data for the frontend JS)
# NOTE: The browser JS should fetch from '/os_questions.json'
@app.route('/static/<path:filename>')
def static_files(filename):
    # This tells Flask to look inside the 'static' folder
    return send_from_directory('static', filename)
@app.route('/os_questions.json')
def serve_questions():
    # os.getcwd() is the root of your project
    return send_from_directory(os.getcwd(), 'os_questions.json')


# LLM OPTIMIZATION ROUTE: Handles the POST request from the client UI
@app.route('/optimize_lesson', methods=['POST'])
def optimize_lesson_route():
    if not os_questions:
        return jsonify({"status": "error", "message": "Question data is missing on the server. Check os_questions.json."}), 500
        
    try:
        # Get data posted from the frontend 
        student_data = request.json
        lesson_topic = student_data.get('topic', 'Process Synchronization')
        
        # Define the student's Skill-Tree profile. 
        # (This could be dynamically generated from quiz_results but is fixed here for simplicity)
        os_student_skills = SkillTree(lesson_topic, {
            "Processes & Threads": 4,      
            "Memory Management": 3,         
            "Concurrency & Sync": 2,        # Low score drives Pertinence in optimization
            "File System & I/O": 4,         
            "OS Fundamentals": 5
        })

        # Run the core multi-agent optimization logic
        best_lp, final_score, skill_tree_summary = run_optimization_core(
            student_skills=os_student_skills,
            lesson_topic=lesson_topic,
            questions_data=os_questions,
            max_iterations=4 # Max iterations for the optimization loop
        )

        return jsonify({
            "status": "success",
            "score": final_score,
            "topic": lesson_topic,
            "best_lesson_plan": best_lp,
            "skill_tree": skill_tree_summary
        })

    except Exception as e:
        app.logger.error(f"Error during optimization: {e}")
        return jsonify({"status": "error", "message": f"Optimization failed due to an internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    if not os_questions:
        print("FATAL: Cannot start server without question data. Check os_questions.json.")
    else:
        # Start the Flask development server on port 5000
        print(f"Flask Server running at http://127.0.0.1:5000")
        app.run(debug=False, port=5000)