# eduplanner/main.py
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, flash
from llm_connector import LLMConnector
from agents.skilltree import SkillTreeAgent
from agents.evaluator import EvaluatorAgent
from agents.optimizer import OptimizerAgent
from agents.analyst import AnalystAgent

# --- Configuration ---
SUBJECT = "Operating Systems"
SUBJECT_SCHEMA_PATH = f"subjects/{SUBJECT.lower().replace(' ', '_')}.json"
LOG_DIR = "data/logs"
MAX_ITERATIONS = 5  # Reduced for faster web demo
CIDDP_SCORE_THRESHOLD = 8.0
QUIZ_ACCURACY_THRESHOLD = 0.8
SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"]

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
# MUST set a secret key for session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_for_local_demo") 

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# --- Helper Functions ---
def load_subject_config(path):
    """Loads the subject configuration file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading subject config: {e}")
        return None

def generate_initial_plan(skill_tree: dict, config: dict) -> str:
    """Generates the first draft of the lesson plan using DeepSeek 70B."""
    prompt = f"""
    Based on this Skill-Tree, where scores are 0-5:
    {json.dumps(skill_tree, indent=2)}
    
    {config['generator_prompt']}
    """
    return LLMConnector.get_expert_response(prompt)

def generate_quiz_questions(topics: list) -> list:
    """Generates quiz questions based on key topics using DeepSeek 70B."""
    topics_str = ", ".join(topics)
    prompt = f"""
    Generate 2 concept-check questions about the following Operating Systems topics: {topics_str}.
    For each question, also provide the correct, concise answer.
    Format the output ONLY as a JSON list of objects, each with "question" and "answer" keys.
    """
    return LLMConnector.get_expert_response(prompt, is_json=True)

def grade_answer(user_answer: str, correct_answer: str) -> bool:
    """Uses LLM to grade the user's answer conceptually."""
    grading_prompt = f"""
    You are an auto-grader. Compare the User Answer against the Correct Answer. 
    Determine if the user's answer is conceptually correct (even if worded differently). 
    Output ONLY 'CORRECT' or 'INCORRECT'.
    
    Correct Answer: {correct_answer}
    User Answer: {user_answer}
    """
    
    grade_result = LLMConnector.get_agent_response(grading_prompt)
    return bool(grade_result and "CORRECT" in grade_result.upper())

def save_session_log(data: dict):
    """Saves the final session data to a JSON log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"{SUBJECT.lower().replace(' ', '_')}_session_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Session log saved to {filename}")

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Initial page to collect student level and start the process."""
    if request.method == 'POST':
        level = request.form.get('level')
        if level in SKILL_LEVELS:
            session.clear()
            session['level'] = level
            session['status_log'] = []
            return redirect(url_for('process'))
        flash("Invalid level selected.")
        
    return render_template('index.html', skill_levels=SKILL_LEVELS, subject=SUBJECT)

@app.route('/process')
def process():
    """Runs the entire iterative LLM pipeline synchronously."""
    level = session.get('level')
    status_log = session.get('status_log')
    
    if not level:
        return redirect(url_for('index'))

    subject_config = load_subject_config(SUBJECT_SCHEMA_PATH)
    if not subject_config:
        status_log.append("ERROR: Could not load subject configuration.")
        return render_template('processing.html', status_log=status_log, finished=True)

    # State variables for the loop
    current_skill_tree = None
    lesson_plan = ""
    final_cidpp_score = 0.0
    iteration = 0
    
    try:
        # --- Step 2: Skill-Tree Generation ---
        skill_tree_agent = SkillTreeAgent(SUBJECT_SCHEMA_PATH)
        status_log.append("STEP 2: [DeepSeek] Building personalized Skill-Tree...")
        current_skill_tree = skill_tree_agent.generate(level)
        status_log.append(f"  > Initial Skill-Tree: {json.dumps(current_skill_tree)}")
        
        # --- Step 3: Initial Lesson Plan Generation ---
        status_log.append("STEP 3: [DeepSeek] Generating initial Lesson Plan draft...")
        lesson_plan = generate_initial_plan(current_skill_tree, subject_config)
        status_log.append(f"  > Draft generated (approx. {len(lesson_plan)} characters).")
        
        # --- Iterative Optimization Loop (Steps 4 & 5) ---
        while iteration < MAX_ITERATIONS:
            iteration += 1
            status_log.append(f"--- OPTIMIZATION ITERATION {iteration} ---")
            
            # Step 4: Evaluator Agent
            status_log.append(f"STEP 4: [OpenAI] Evaluating plan...")
            feedback = EvaluatorAgent.evaluate(lesson_plan, current_skill_tree)
            
            if not feedback:
                status_log.append("  > [ERROR] Evaluation failed. Stopping loop.")
                break
                
            final_cidpp_score = feedback.get("Average", 0.0)
            status_log.append(f"  > Evaluator Score: {final_cidpp_score:.1f}. Disadvantages: {feedback.get('Disadvantages')}")

            # Check stopping criteria based on score
            if final_cidpp_score >= CIDDP_SCORE_THRESHOLD:
                status_log.append(f"  > [SUCCESS] CIDDP score threshold ({CIDDP_SCORE_THRESHOLD}) met. Optimizing once more for robustness.")
                
            if iteration < MAX_ITERATIONS or final_cidpp_score < CIDDP_SCORE_THRESHOLD:
                # Step 5: Optimizer Agent
                status_log.append(f"STEP 5: [OpenAI] Optimizing plan...")
                lesson_plan = OptimizerAgent.optimize(lesson_plan, feedback)
                status_log.append("  > Plan optimized successfully.")
                
            if final_cidpp_score >= CIDDP_SCORE_THRESHOLD:
                break
                
        # --- Step 6: Analyst Agent ---
        status_log.append("STEP 6: [OpenAI] Analyzing for common pitfalls...")
        error_dict = AnalystAgent.analyze(lesson_plan)
        error_tips = AnalystAgent.format_errors(error_dict)
        final_lesson_plan = lesson_plan + error_tips
        status_log.append(f"  > Analyst added {len(error_dict)} topics of common mistakes.")
        
        # --- Prepare for Quiz (Step 7) ---
        questions = generate_quiz_questions(subject_config.get("quiz_topics", []))
        session['questions'] = questions
        session['final_lesson_plan'] = final_lesson_plan
        session['final_cidpp_score'] = final_cidpp_score
        session['current_skill_tree'] = current_skill_tree
        session['iterations_run'] = iteration
        
        # Redirect to the quiz
        return redirect(url_for('quiz'))

    except Exception as e:
        status_log.append(f"FATAL ERROR in pipeline: {e}")
        return render_template('processing.html', status_log=status_log, finished=True)

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """Handles the user interaction for the quiz questions."""
    questions = session.get('questions')
    
    if not questions:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        correct_count = 0
        total_questions = len(questions)
        
        for i, q_item in enumerate(questions):
            user_answer = request.form.get(f'q{i}')
            is_correct = grade_answer(user_answer, q_item['answer'])
            if is_correct:
                correct_count += 1
                
        quiz_accuracy = correct_count / total_questions
        session['quiz_accuracy'] = quiz_accuracy
        
        # Redirect to the final plan summary
        return redirect(url_for('final_plan'))
        
    # GET request: Display the quiz form
    return render_template('quiz.html', questions=questions)


@app.route('/final_plan')
def final_plan():
    """Displays the final summary, plan, and performs logging."""
    quiz_accuracy = session.get('quiz_accuracy', 0.0)
    final_cidpp_score = session.get('final_cidpp_score', 0.0)
    final_lesson_plan = session.get('final_lesson_plan', "No plan found.")
    current_skill_tree = session.get('current_skill_tree', {})
    iterations_run = session.get('iterations_run', 0)

    # Step 7 (Post-Quiz Update)
    updated_skill_tree = SkillTreeAgent.update_tree(current_skill_tree, quiz_accuracy)

    # Step 8/9: Stopping Criteria and Logging
    final_status = "Incomplete/Needs Practice"
    next_step = "The plan quality is high, but your quiz performance suggests focusing on a few more practice problems before proceeding."
    
    if final_cidpp_score >= CIDDP_SCORE_THRESHOLD and quiz_accuracy >= QUIZ_ACCURACY_THRESHOLD:
        final_status = "Module Completed"
        next_step = "You've successfully completed this module. You are now ready for the next set of Intermediate topics or the Computer Networks module."

    session_data = {
        "timestamp": datetime.now().isoformat(),
        "student_level": session.get('level', 'N/A'),
        "subject": SUBJECT,
        "iterations_run": iterations_run,
        "final_cidpp": final_cidpp_score,
        "quiz_accuracy": quiz_accuracy,
        "final_skill_tree": updated_skill_tree,
        "final_lesson_plan_sample": final_lesson_plan[:500] + "...", 
    }
    save_session_log(session_data)
    
    return render_template(
        'final_plan.html',
        status=final_status,
        score=f"{final_cidpp_score:.1f}",
        accuracy=f"{quiz_accuracy*100:.1f}",
        next_step=next_step,
        skill_tree=updated_skill_tree,
        lesson_plan=final_lesson_plan
    )

if __name__ == '__main__':
    # Flask will automatically look for templates in the 'templates' folder
    # Host on localhost:5000 as requested
    app.run(debug=True, host='127.0.0.1', port=5000)