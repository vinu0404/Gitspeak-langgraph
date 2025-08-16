import os
import ast
import git
import nbformat
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Global retriever variable
retriever = None


def clone_repo(repo_url):
    """Clone a Git repository to local directory"""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    if not os.path.exists("repos"):
        os.makedirs("repos", exist_ok=True)
    local_path = os.path.join("repos", repo_name)
    if not os.path.exists(local_path):
        git.Repo.clone_from(repo_url, local_path)
    return local_path


def extract_code_units(file_path):
    """Extract code units (functions, classes) from various file types"""
    units = []
    
    if file_path.endswith(".py"):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    chunk = "\n".join(code.splitlines()[start_line-1:end_line])
                    metadata = {"file": file_path, "name": node.name, "type": type(node).__name__}
                    units.append((chunk, metadata))
        except SyntaxError:
            metadata = {"file": file_path, "name": "unknown", "type": "file"}
            units.append((code, metadata))
    
    elif file_path.endswith(".ipynb"):
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code = cell.source
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            chunk = code
                            metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "cell_index": notebook.cells.index(cell)}
                            units.append((chunk, metadata))
                except SyntaxError:
                    metadata = {"file": file_path, "name": "unknown", "type": "cell", "cell_index": notebook.cells.index(cell)}
                    units.append((code, metadata))
    
    elif file_path.endswith((
        ".c", ".cpp",
        ".js", ".ts", ".tsx", ".jsx",
        ".go",
        ".java",
        ".rb",
        ".php",
        ".swift",
        ".sh", ".bash",
        ".ps1",
        ".r", ".R",
        ".yaml", ".yml",
        ".json",
        ".xml",
        ".html", ".htm",
        ".css", ".scss", ".sass",
        ".sql",
        ".dockerfile", ".Dockerfile",
        ".md", ".markdown",
        ".makefile", ".Makefile", ".mk"
    )):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        metadata = {"file": file_path, "name": os.path.basename(file_path), "type": "file"}
        units.append((code, metadata))
    
    return units


def build_retriever(repo_url):
    """Build vector store retriever from repository"""
    global retriever
    
    local_path = clone_repo(repo_url)
    code_units = []
    
    for root, _, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            units = extract_code_units(file_path)
            code_units.extend(units)
    
    if not code_units:
        raise ValueError("No code units found in the repository")
    
    # Use OpenAI's best embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
    )
    
    chunks = [chunk for chunk, _ in code_units]
    metadata = [meta for _, meta in code_units]
    
    # Create vector store
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata
    )
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    return retriever


# LLM Setup - Use GPT-4o for all operations
rag_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.1
)

basic_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.1
)

template = """Answer the question based on the following context and the Chat history. Especially take the latest question into consideration:

Chat history: {history}

Context: {context}

Question: {question}

### Instructions:
1. Analyze the code snippets to answer the query if they contain relevant information.
2. Provide a clear, technical explanation, referencing specific parts of the snippets (e.g., function names, key logic) when applicable.
3. If the query involves a function or class, describe its purpose, inputs, outputs, and key logic based on the snippets.
4. If the snippets lack sufficient information, use your general technical expertise to answer, explaining how the query relates to typical programming practices or the likely context of the codebase.
5. Keep the response focused, technical, and complete, addressing all parts of the query.
6. For questions about files (like README.md, main.py, etc.), provide detailed explanations based on the available content.
7. Always attempt to provide a helpful answer even if the context is limited.
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = prompt | rag_llm


# State and Model Definitions
class AgentState(TypedDict):
    messages: List[BaseMessage]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: str
    rephrase_count: int
    question: HumanMessage
    code: list


class OffTopic(BaseModel):
    answer: str = Field(description="Question is from specified topic? If yes -> 'yes' if not -> 'no'")


class GradeCodeChunk(BaseModel):
    relevance: str = Field(description="Relevance of the code chunk to the question, if 'relevant' -> 'relevant' else 'not relevant'")


# Node Functions
def question_rewriter(state: AgentState):
    """Rewrite questions to be more specific and on-topic"""
    state["code"] = []
    state["rephrased_question"] = ""
    state["rephrase_count"] = 0
    state["on_topic"] = "no"
    state["proceed_to_generate"] = "no"
    
    if state.get("messages") is None:
        state["messages"] = []
    
    question_content = state["question"].content if hasattr(state["question"], 'content') else str(state["question"])
    
    if len(state["messages"]) > 1:
        conversation_history = state["messages"][:-1]
        messages = [
            SystemMessage(content="""You are a helpful assistant that rewrites questions to be more specific and effective for code repository analysis.

Guidelines for rewriting:
1. Keep the core intent of the question
2. Make it more specific to code/repository analysis if needed
3. Ensure it's clear what type of information is being requested
4. For file-specific questions (like "What does main.py do?"), keep them focused on that file
5. For general questions, make them more specific to the codebase context
6. Don't change questions that are already clear and specific

Examples:
- "What does this do?" → "What does the main functionality of this codebase do?"
- "Explain the code" → "Explain the key functions and classes in this codebase"
- "What is in README?" → "What is written in the README.md file?"
"""),
            HumanMessage(content=f"Original question: {question_content}\n\nRewrite this question to be more specific and effective for code analysis:")
        ]
        messages.extend(conversation_history)
        
        # Fixed: Pass messages list directly to LLM
        response = basic_llm.invoke(messages)
        better_question = response.content.strip()
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = question_content
    
    return state


def question_classifier(state: AgentState):
    """Classify if question is code-related"""
    messages = [
        SystemMessage(content="""You are a helpful assistant that classifies questions about code repositories.

Your job is to determine if a question is related to programming, code, or repository content.

ALWAYS respond with 'yes' for questions about:
- Code files (main.py, app.py, config.py, etc.)
- Repository files (README.md, requirements.txt, package.json, etc.) 
- Functions, classes, or code structure
- How code works or what it does
- Code review, explanation, or analysis
- Project documentation or setup
- Any technical aspect of the repository

ONLY respond with 'no' for questions that are:
- Completely unrelated to programming or the repository
- Personal questions about people
- General non-technical topics (weather, sports, etc.)

Examples of 'yes' questions:
- "What does main.py do?"
- "Explain the key classes"
- "What's in the README?"
- "How does this function work?"
- "What files are in this repo?"

Examples of 'no' questions:
- "What's the weather like?"
- "Tell me about your day"
- "What's your favorite color?"

When in doubt, answer 'yes' to be helpful.
"""),
        HumanMessage(content=f"Question to classify: '{state['rephrased_question']}'")
    ]
    
    structured_llm = basic_llm.with_structured_output(OffTopic)
    response = structured_llm.invoke(messages)
    state["on_topic"] = response.answer.strip().lower()
    print(f"🎯 Question classification: '{state['rephrased_question']}' -> {state['on_topic']}")
    return state


def on_topic_router(state: AgentState):
    """Route based on whether question is on-topic"""
    print(f"🔍 Classification result: {state['on_topic']}")
    if state["on_topic"] == "yes":
        return "retriever"
    else:
        return "off_topic"


def retriever_node(state: AgentState):
    """Retrieve relevant code chunks"""
    if retriever is None:
        raise ValueError("Retriever not initialized. Please run build_retriever() first.")
    
    code_units = retriever.invoke(state["rephrased_question"])
    state["code"] = code_units
    return state


def relevant_code_chunk(state: AgentState):
    """Filter code chunks for relevance"""
    system_message = SystemMessage(content="""You are a code chunk relevance classifier. Your job is to determine if a code chunk is relevant to answering the user's question.

CLASSIFY AS 'relevant' if the code chunk:
- Contains functions, classes, or variables mentioned in the question
- Shows implementation details that help answer the question
- Contains file content (like README.md) that the user is asking about
- Has documentation, comments, or structure that relates to the question
- Contains configuration, setup, or architectural information relevant to the query
- Shows error handling, imports, or dependencies that relate to the question

CLASSIFY AS 'not relevant' ONLY if the code chunk:
- Has absolutely no connection to the question
- Contains completely unrelated functionality

When in doubt, classify as 'relevant' to provide comprehensive answers.
""")
    structured_llm = basic_llm.with_structured_output(GradeCodeChunk)
    relevant_code_chunks = []
    
    for code in state["code"]:
        human_message = HumanMessage(content=f"Code chunk to evaluate:\n{code}\n\nQuestion: {state['rephrased_question']}\n\nIs this code chunk relevant to answering the question?")
        messages = [system_message, human_message]
        response = structured_llm.invoke(messages)
        if response.relevance.strip().lower() == "relevant":
            relevant_code_chunks.append(code)
    
    state["code"] = relevant_code_chunks
    
    # Set proceed_to_generate based on whether we have relevant code
    if len(relevant_code_chunks) > 0:
        state["proceed_to_generate"] = "yes"
    else:
        state["proceed_to_generate"] = "no"
    
    return state


def router(state: AgentState):
    """Route to next step based on state"""
    if state["rephrase_count"] > 2:
        return "cannot_answer"
    elif state["proceed_to_generate"] == "yes":
        return "generate_answer"
    else:
        return "refine_question"


def refine_question(state: AgentState):
    """Refine the question to improve retrieval results"""
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        return state
    
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    messages = [system_message, human_message]
    response = basic_llm.invoke(messages)
    refined_question = response.content.strip()
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state


def generate_answer(state: AgentState):
    """Generate final answer using RAG chain"""
    history = state["messages"]
    code_units = state["code"]
    rephrased_question = state["rephrased_question"]
    
    # Format history as string for template
    history_str = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_str += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"AI: {msg.content}\n"
        else:
            history_str += f"System: {msg.content}\n"
    
    response = rag_chain.invoke({
        "history": history_str, 
        "context": code_units, 
        "question": rephrased_question
    })
    generation = response.content.strip()
    state["messages"].append(AIMessage(content=generation))
    return state


def cannot_answer(state: AgentState):
    """Handle cases where we cannot answer"""
    state["messages"].append(AIMessage(content="I'm sorry, I cannot answer that question with the available code context."))
    return state


def off_topic(state: AgentState):
    """Handle off-topic questions"""
    state["messages"].append(AIMessage(content="This question is off-topic. Please ask a question related to code review, generation, explanation, summarization, or about the repository."))
    return state


# Workflow Setup
checkpoint = MemorySaver()

def create_workflow():
    """Create and compile the workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("relevant_code_chunk", relevant_code_chunk)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("cannot_answer", cannot_answer)
    workflow.add_node("off_topic", off_topic)
    
    # Add edges
    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges("question_classifier", on_topic_router, {
        "retriever": "retriever",
        "off_topic": "off_topic"
    })
    workflow.add_edge("retriever", "relevant_code_chunk")
    workflow.add_conditional_edges("relevant_code_chunk", router, {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer"
    })
    workflow.add_edge("refine_question", "retriever")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("cannot_answer", END)
    workflow.set_entry_point("question_rewriter")
    
    return workflow.compile(checkpointer=checkpoint)


# Main execution functions
def setup_system(repo_url):
    """Initialize the entire RAG system"""
    print(f"Setting up RAG system for repository: {repo_url}")
    print("This may take a few minutes depending on repository size...")
    
    try:
        build_retriever(repo_url)
        print("✅ System setup complete!")
        return True
    except Exception as e:
        print(f"❌ Error setting up system: {e}")
        return False


def ask_question(question_text, graph, thread_id=1, conversation_history=None):
    """Ask a single question to the system with streaming"""
    if conversation_history is None:
        conversation_history = []
    
    input_data = {
        "question": HumanMessage(content=question_text),
        "messages": conversation_history.copy(),
        "on_topic": "",
        "rephrased_question": "",
        "proceed_to_generate": "",
        "rephrase_count": 0,
        "code": []
    }
    
    try:
        print("🔄 Processing your question...")
        
        # Stream the execution
        final_state = None
        for event in graph.stream(
            input=input_data,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        ):
            final_state = event
            
            # Show progress for different nodes
            current_node = None
            if hasattr(event, 'get'):
                # Try to determine current processing stage based on state
                if event.get("on_topic") == "":
                    current_node = "Analyzing question type"
                elif event.get("on_topic") in ["yes", "no"] and not event.get("code"):
                    current_node = "Retrieving relevant code"
                elif event.get("code") and event.get("proceed_to_generate") == "":
                    current_node = "Evaluating code relevance"
                elif event.get("proceed_to_generate") == "yes":
                    current_node = "Generating answer"
            
            if current_node:
                print(f"⚡ {current_node}...")
        
        if final_state and final_state.get("messages"):
            answer = final_state["messages"][-1].content
            return answer, final_state["messages"]
        else:
            return "No answer generated.", []
            
    except Exception as e:
        return f"Error processing question: {e}", []


def ask_question_streaming(question_text, graph, thread_id=1, conversation_history=None):
    """Ask a question with real-time streaming output"""
    if conversation_history is None:
        conversation_history = []
    
    input_data = {
        "question": HumanMessage(content=question_text),
        "messages": conversation_history.copy(),
        "on_topic": "",
        "rephrased_question": "",
        "proceed_to_generate": "",
        "rephrase_count": 0,
        "code": []
    }
    
    try:
        print("🔄 Processing your question...")
        print("📊 Progress:")
        
        step_count = 0
        final_state = None
        
        for event in graph.stream(
            input=input_data,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        ):
            step_count += 1
            final_state = event
            
            # Determine current step based on what we have in the state
            status = "Processing"
            
            # Check if we have a rephrased question but no classification yet
            if event.get("rephrased_question") and event.get("on_topic") == "":
                status = "🔍 Classifying question"
            # Check if question was classified as off-topic
            elif event.get("on_topic") == "no":
                status = "❌ Question is off-topic"
            # Check if question was classified as on-topic but no code retrieved yet
            elif event.get("on_topic") == "yes" and not event.get("code"):
                status = "📚 Retrieving code chunks"
            # Check if we have code but haven't decided whether to proceed
            elif event.get("code") and event.get("proceed_to_generate") == "":
                status = "🎯 Filtering relevant code"
            # Check if we need to refine the question
            elif event.get("proceed_to_generate") == "no" and event.get("rephrase_count", 0) > 0:
                status = f"🔄 Refining query (attempt {event.get('rephrase_count', 0)})"
            # Check if we're ready to generate
            elif event.get("proceed_to_generate") == "yes":
                status = "✨ Generating answer"
            # Check if we have a final message
            elif event.get("messages") and len(event["messages"]) > 0:
                last_message = event["messages"][-1].content.lower()
                if "off-topic" in last_message:
                    status = "❌ Completed - Off-topic"
                elif "cannot answer" in last_message:
                    status = "❌ Completed - Cannot answer"
                else:
                    status = "✅ Answer generated"
            
            print(f"   Step {step_count}: {status}")
        
        print("\n🎉 Processing complete!")
        
        if final_state and final_state.get("messages"):
            answer = final_state["messages"][-1].content
            return answer, final_state["messages"]
        else:
            return "No answer generated.", []
            
    except Exception as e:
        return f"Error processing question: {e}", []


def interactive_mode():
    """Run the system in interactive mode with streaming"""
    print("\n🎯 Interactive Code Analysis Mode")
    print("=" * 50)
    
    # Get repository URL
    repo_url = input("Enter GitHub repository URL: ").strip()
    if not repo_url:
        print("No repository URL provided. Exiting.")
        return
    
    # Setup system
    if not setup_system(repo_url):
        return
    
    # Create workflow
    graph = create_workflow()
    
    print("\n✅ System ready! Ask questions about the code.")
    print("💡 Type 'quit' to exit")
    print("🎛️  Type 'stream' to toggle between streaming/non-streaming mode")
    print("-" * 50)
    
    conversation_history = []
    thread_id = 1
    streaming_mode = True
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.lower() == 'stream':
            streaming_mode = not streaming_mode
            mode_text = "ON" if streaming_mode else "OFF"
            print(f"🎛️  Streaming mode: {mode_text}")
            continue
        
        if not question:
            continue
        
        if streaming_mode:
            print("\n🚀 Processing with streaming...")
            answer, updated_history = ask_question_streaming(question, graph, thread_id, conversation_history)
        else:
            print("🤔 Processing...")
            answer, updated_history = ask_question(question, graph, thread_id, conversation_history)
        
        print(f"\n🤖 Answer: {answer}")

        # Update conversation history
        conversation_history = updated_history
        print("-" * 50)


def demo_mode():
    """Run with sample questions with streaming"""
    print("\n🎯 Demo Mode")
    print("=" * 50)
    
    # Example repository (you can change this)
    repo_url = input("Enter GitHub repository URL (or press Enter for demo): ").strip()
    if not repo_url:
        repo_url = "https://github.com/vinu0404/GitSpeak"  
    
    if not setup_system(repo_url):
        return
    
    graph = create_workflow()
    
    sample_questions = [
        "What does the main.py file do?",
        "Can you explain the key classes in this repository?",
        "What are the main functions defined in this codebase?",
        "Can you review the error handling in this code?",
        "What is written in the README.md file?"
    ]
    
    print(f"\n🚀 Running demo questions on repository: {repo_url}")
    print("=" * 70)
    
    # Ask user for streaming preference
    use_streaming = input("\nUse streaming mode? (y/n, default: y): ").strip().lower()
    use_streaming = use_streaming != 'n'
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        if use_streaming:
            answer, _ = ask_question_streaming(question, graph, thread_id=i)
        else:
            print("🤔 Processing...")
            answer, _ = ask_question(question, graph, thread_id=i)
            
        print(f"\n🤖 Answer: {answer}")
        print("-" * 70)


if __name__ == "__main__":
    print("🔧 OpenAI RAG Code Analysis System")
    print("=" * 40)
    print("Choose mode:")
    print("1. Interactive mode")
    print("2. Demo mode")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        demo_mode()
    else:
        print("Invalid choice. Starting interactive mode...")
        interactive_mode()