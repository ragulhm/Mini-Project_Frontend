# eduplanner/ui/display.py
import json
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich Console (shared instance)
console = Console()

def display_welcome(subject: str):
    """Displays the welcome message."""
    console.rule("[bold green]Welcome to EduPlanner 🔥[/bold green]")
    console.print(f"Starting with [bold yellow]{subject}[/bold yellow] module.")

def get_student_level(skill_levels: list) -> str:
    """Prompts the user to select their skill level using inquirer."""
    questions = [
        inquirer.List(
            'level',
            message="Please select your skill level",
            choices=skill_levels,
        ),
    ]
    answers = inquirer.prompt(questions)
    return answers['level']

def display_skill_tree(skill_tree: dict, title: str):
    """Displays the skill tree in a readable table format."""
    console.print("\n" + "="*80)
    table = Table(title=f"[bold magenta]{title}[/bold magenta]")
    table.add_column("Skill Node", style="cyan")
    table.add_column("Score (0-5)", style="yellow", justify="right")
    
    for node, score in skill_tree.items():
        table.add_row(node, f"{score}/5")
        
    console.print(table)
    console.print("="*80)

def display_pipeline_step(step_number: int, model_name: str, task_description: str):
    """Displays which agent/model is currently running a task."""
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    console.print(f"\n[bold blue]Pipeline Step {step_number}:[/bold blue] [cyan]{model_short}[/cyan] is {task_description}...")

def display_optimization_iteration(iteration: int):
    """Displays the start of a new optimization iteration."""
    console.rule(f"[bold yellow]*** OPTIMIZATION ITERATION {iteration} ***[/bold yellow]")

def display_evaluator_summary(feedback: dict):
    """Displays the results from the Evaluator Agent."""
    avg_score = feedback.get("Average", 0.0)
    
    console.print(Panel(
        f"Average CIDDP Score: [bold magenta]{avg_score:.1f}[/bold magenta] / 10.0\n"
        f"Strengths: [green]{', '.join(feedback.get('Advantages', []))}[/green]\n"
        f"Disadvantages (Needs Fixing): [red]{', '.join(feedback.get('Disadvantages', []))}[/red]",
        title="[bold blue]Evaluator Agent Summary[/bold blue]",
        border_style="yellow"
    ))

def display_optimization_status(success: bool, iteration: int, max_iterations: int):
    """Displays the result of an optimization check."""
    if success:
        console.print("[bold green]✅ Plan optimized successfully (Draft {iteration+1} generated).[/bold green]")
    elif iteration >= max_iterations:
        console.print("[bold red]Max iterations reached. Finalizing the current plan.[/bold red]")
    else:
        # For the threshold check success
        console.print("[bold green]✅ CIDDP score threshold met. Moving to final analysis and quiz.[/bold green]")

def display_analyst_tips(error_tips: str):
    """Displays the common mistakes (Analyst Agent output)."""
    if error_tips:
        console.print(Panel(
            error_tips.strip(),
            title="[bold red]🧠 AVOID THESE MISTAKES (Analyst Tips)[/bold red]",
            border_style="red"
        ))

def prompt_quiz_question(i: int, q_text: str) -> str:
    """Prompts the user for a quiz answer."""
    questions_prompt = [
        inquirer.Text(f'q{i}', message=f"Q{i}: {q_text}")
    ]
    answers = inquirer.prompt(questions_prompt)
    return answers[f'q{i}']

def display_quiz_grade(is_correct: bool, correct_answer: str):
    """Displays the result of the grading for a single question."""
    if is_correct:
        console.print("   [bold green]RESULT: CORRECT ✅[/bold green]")
    else:
        console.print(f"   [bold red]RESULT: INCORRECT ❌[/bold red] (Correct: {correct_answer})")
    console.print("-" * 50)

def display_quiz_summary(accuracy: float):
    """Displays the overall quiz summary."""
    console.print(Panel(
        f"[bold white]Overall Quiz Accuracy: [yellow]{accuracy*100:.1f}%[/yellow][/bold white]", 
        title="[bold blue]Quiz Summary[/bold blue]", 
        border_style="blue"
    ))

def display_final_summary(status: str, score: float, accuracy: float, next_step: str):
    """Displays the final session summary."""
    console.rule("[bold green]SESSION COMPLETED[/bold green]")
    console.print(Panel(
        f"[bold white]Status:[/bold white] [green]{status}[/green]\n"
        f"[bold white]Final Plan Score (CIDDP):[/bold white] [yellow]{score:.1f}[/yellow]\n"
        f"[bold white]Quiz Performance:[/bold white] [yellow]{accuracy*100:.1f}%[/yellow]\n"
        f"[bold white]Next Step Suggestion:[/bold white] {next_step}",
        title="[bold green]Final Summary[/bold green]",
        border_style="green"
    ))

def prompt_display_final_plan(final_plan: str):
    """Prompts the user and displays the final plan if confirmed."""
    if inquirer.confirm('Do you want to see the full, optimized lesson plan now?', default=True):
        console.rule("[bold blue]Full Optimized Lesson Plan[/bold blue]")
        console.print(final_plan)