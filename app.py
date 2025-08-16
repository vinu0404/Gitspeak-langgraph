import os
import ast
import git
import json
import pickle
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
PERSISTENT_DIR = Path("gitspeak_persistent")
PERSISTENT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(PERSISTENT_DIR / "checkpoints.sqlite")
GLOBAL_RETRIEVERS = {}
GLOBAL_SESSIONS = {}
rag_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=0.1,
    streaming=True  
)
basic_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=0.1,
    streaming=True 
)
template = """You are GitSpeak, an AI assistant specialized in code repository analysis. 

Provide well-structured, detailed responses using proper markdown formatting.

Chat history: {history}
Context: {context}
Question: {question}

### Instructions:
1. **Structure your response clearly** using markdown headers (##, ###)
2. **Use code blocks** with proper syntax highlighting for code examples
3. **Use bullet points or numbered lists** for multiple items
4. **Bold important concepts** and *italicize* file names
5. **Quote specific code snippets** when referencing them
6. **Provide clear explanations** with technical depth
7. **Create tables** when comparing multiple items
8. **Add horizontal rules (----)** to separate sections when needed
9. **Always provide actionable insights** and next steps when relevant

### Response Format Examples:
- Use `inline code` for variable names, function names, and short snippets
- Use ```python code blocks``` for longer code examples
- Use > blockquotes for important notes or warnings
- Use **bold** for key concepts and *italics* for emphasis

Provide a comprehensive, well-formatted response that would be helpful for a developer working with this codebase.
"""

prompt = ChatPromptTemplate.from_template(template)
def create_streaming_rag_chain():
    return prompt | rag_llm

rag_chain = create_streaming_rag_chain()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: str
    rephrase_count: int
    question: HumanMessage
    code: list
    repo_hash: str
    conversation_history: List[BaseMessage]

class OffTopic(BaseModel):
    answer: str = Field(description="Is the question about code/repository content? Answer 'yes' or 'no' only.")

class BatchCodeGrading(BaseModel):
    relevance_scores: List[str] = Field(
        description="List of relevance scores for each code chunk. Each score should be 'relevant' or 'not relevant'"
    )

class GradeCodeChunk(BaseModel):
    relevance: str = Field(description="Relevance of the code chunk to the question, if 'relevant' -> 'relevant' else 'not relevant'")


def get_repo_hash(repo_url: str) -> str:
    """Generate unique hash for repository URL"""
    return hashlib.md5(repo_url.encode()).hexdigest()


def save_session_data(session_id: str, data: Dict[str, Any]):
    """Save session data to persistent storage"""
    session_file = PERSISTENT_DIR / f"session_{session_id}.json"
    try:
        # Convert messages to serializable format
        serializable_data = data.copy()
        if 'conversation_history' in serializable_data:
            history = []
            for msg in serializable_data['conversation_history']:
                if hasattr(msg, 'content'):
                    msg_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                    history.append({'type': msg_type, 'content': msg.content})
            serializable_data['conversation_history'] = history
        
        serializable_data['last_updated'] = datetime.now().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    except Exception as e:
        print(f"Error saving session data: {e}")


def load_session_data(session_id: str) -> Dict[str, Any]:
    """Load session data from persistent storage"""
    session_file = PERSISTENT_DIR / f"session_{session_id}.json"
    try:
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
                
                if 'conversation_history' in data:
                    history = []
                    for msg_data in data['conversation_history']:
                        if msg_data['type'] == 'human':
                            history.append(HumanMessage(content=msg_data['content']))
                        else:
                            history.append(AIMessage(content=msg_data['content']))
                    data['conversation_history'] = history
                return data
    except Exception as e:
        print(f"Error loading session data: {e}")
    return {}


def save_retriever_data(repo_hash: str, vector_store, repo_name: str, repo_url: str):
    """Save vector store and metadata"""
    try:
        retriever_dir = PERSISTENT_DIR / f"retriever_{repo_hash}"
        retriever_dir.mkdir(exist_ok=True)
        
        # Save FAISS vector store
        vector_store.save_local(str(retriever_dir / "faiss_store"))
        
        # Save metadata
        metadata = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "created_at": datetime.now().isoformat()
        }
        with open(retriever_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error saving retriever: {e}")


def load_retriever_data(repo_hash: str):
    """Load vector store and metadata"""
    try:
        retriever_dir = PERSISTENT_DIR / f"retriever_{repo_hash}"
        
        if not retriever_dir.exists():
            return None, None, None
            
        # Load metadata first
        metadata_file = retriever_dir / "metadata.json"
        if not metadata_file.exists():
            return None, None, None
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load FAISS vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(
            str(retriever_dir / "faiss_store"), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})
        
        return retriever, metadata["repo_name"], metadata["repo_url"]
        
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None, None, None


def find_existing_repositories() -> List[Dict[str, str]]:
    """Find all existing repository data"""
    existing_repos = []
    if PERSISTENT_DIR.exists():
        for retriever_dir in PERSISTENT_DIR.glob("retriever_*"):
            try:
                metadata_file = retriever_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    repo_hash = retriever_dir.name.replace('retriever_', '')
                    existing_repos.append({
                        'repo_name': metadata['repo_name'],
                        'repo_url': metadata['repo_url'],
                        'repo_hash': repo_hash,
                        'created_at': metadata.get('created_at', 'Unknown')
                    })
            except Exception:
                continue
    return existing_repos


def extract_code_units(file_path):
    """Extract code units from various file types"""
    units = []
    
    try:
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
        
        elif file_path.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".xml", ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".cpp", ".c", ".h", ".rs")):
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            metadata = {"file": file_path, "name": os.path.basename(file_path), "type": "file"}
            units.append((code, metadata))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return units


async def build_retriever_from_repo(repo_url: str) -> tuple:
    """Build retriever from repository URL"""
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_hash = get_repo_hash(repo_url)
        
        # Check if already exists in global cache
        if repo_hash in GLOBAL_RETRIEVERS:
            cached_data = GLOBAL_RETRIEVERS[repo_hash]
            return cached_data['retriever'], cached_data['repo_name'], repo_hash
        
        # Try to load from disk
        existing_retriever, existing_name, existing_url = load_retriever_data(repo_hash)
        if existing_retriever:
            # Cache in memory
            GLOBAL_RETRIEVERS[repo_hash] = {
                'retriever': existing_retriever,
                'repo_name': existing_name,
                'repo_url': existing_url
            }
            await cl.Message(
                content=f"**Found cached data for `{existing_name}`!**",
                author="GitSpeak"
            ).send()
            return existing_retriever, existing_name, repo_hash
        
        # Build new retriever
        await cl.Message(
            content=f"**Processing new repository: `{repo_name}`**",
            author="GitSpeak"
        ).send()
        
        repos_dir = Path("repos")
        repos_dir.mkdir(exist_ok=True)
        
        local_path = repos_dir / repo_name
        if not local_path.exists():
            await cl.Message(
                content="**Cloning repository...**",
                author="GitSpeak"
            ).send()
            git.Repo.clone_from(repo_url, str(local_path))
        
        # Extract code units
        await cl.Message(
            content="**Extracting and analyzing code structure...**",
            author="GitSpeak"
        ).send()
        
        code_units = []
        file_count = 0
        
        for root, _, files in os.walk(local_path):
            for file in files:
                if file_count > 1000: 
                    break
                file_path = os.path.join(root, file)
                units = extract_code_units(file_path)
                if units:
                    code_units.extend(units)
                    file_count += 1
        
        if not code_units:
            raise ValueError("No analyzable code units found in the repository")
        
        # Build vector store
        await cl.Message(
            content=f"**Building knowledge base from {len(code_units)} code units...**",
            author="GitSpeak"
        ).send()
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        
        chunks = [chunk for chunk, _ in code_units]
        metadata = [meta for _, meta in code_units]
        
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadata)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})
        
        # Save to disk and cache in memory
        save_retriever_data(repo_hash, vector_store, repo_name, repo_url)
        GLOBAL_RETRIEVERS[repo_hash] = {
            'retriever': retriever,
            'repo_name': repo_name,
            'repo_url': repo_url
        }
        
        return retriever, repo_name, repo_hash
        
    except Exception as e:
        raise Exception(f"Error building retriever: {str(e)}")


# LangGraph Node Functions
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
    
    if len(state.get("conversation_history", [])) > 1:
        conversation_history = state["conversation_history"][:-1]
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

When in doubt, answer 'yes' to be helpful.
"""),
        HumanMessage(content=f"Question to classify: '{state['rephrased_question']}'")
    ]
    
    structured_llm = basic_llm.with_structured_output(OffTopic)
    response = structured_llm.invoke(messages)
    state["on_topic"] = response.answer.strip().lower()
    print(f"Question classification: '{state['rephrased_question']}' -> {state['on_topic']}")
    return state


def on_topic_router(state: AgentState):
    """Route based on whether question is on-topic"""
    print(f"Classification result: {state['on_topic']}")
    if state["on_topic"] == "yes":
        return "retriever"
    else:
        return "off_topic"


def retriever_node(state: AgentState):
    """Retrieve relevant code chunks"""
    repo_hash = state.get("repo_hash")
    if repo_hash not in GLOBAL_RETRIEVERS:
        raise ValueError("Retriever not found for current repository.")
    
    retriever = GLOBAL_RETRIEVERS[repo_hash]['retriever']
    code_units = retriever.invoke(state["rephrased_question"])
    state["code"] = code_units
    return state


def batch_relevant_code_chunk(state: AgentState):
    """Filter code chunks for relevance using batch processing"""
    system_message = SystemMessage(content="""You are a code chunk relevance classifier. Your job is to determine if code chunks are relevant to answering the user's question.

For each code chunk, classify as 'relevant' if the code chunk:
- Contains functions, classes, or variables mentioned in the question
- Shows implementation details that help answer the question
- Contains file content (like README.md) that the user is asking about
- Has documentation, comments, or structure that relates to the question
- Contains configuration, setup, or architectural information relevant to the query
- Shows error handling, imports, or dependencies that relate to the question

Classify as 'not relevant' ONLY if the code chunk:
- Has absolutely no connection to the question
- Contains completely unrelated functionality

When in doubt, classify as 'relevant' to provide comprehensive answers.

Return a list of relevance scores in the same order as the code chunks provided.
""")
    code_chunks_text = ""
    for i, code in enumerate(state["code"]):
        code_chunks_text += f"=== Code Chunk {i+1} ===\n{code}\n\n"
    
    human_message = HumanMessage(
        content=f"Code chunks to evaluate:\n{code_chunks_text}\n\nQuestion: {state['rephrased_question']}\n\nProvide relevance scores for each code chunk in order."
    )
    
    messages = [system_message, human_message]
    structured_llm = basic_llm.with_structured_output(BatchCodeGrading)
    response = structured_llm.invoke(messages)
    relevant_code_chunks = []
    for i, score in enumerate(response.relevance_scores):
        if i < len(state["code"]) and score.strip().lower() == "relevant":
            relevant_code_chunks.append(state["code"][i])
    
    state["code"] = relevant_code_chunks
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


async def generate_answer_streaming(state: AgentState):
    history = state.get("conversation_history", [])
    code_units = state["code"]
    rephrased_question = state["rephrased_question"]
    
    # Format conversation history
    history_str = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_str += f"**User:** {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            history_str += f"**GitSpeak:** {msg.content[:200]}...\n\n"
    
    # Create a new streaming message
    streaming_msg = cl.Message(content="", author="GitSpeak")
    await streaming_msg.send()
    
    # Stream the response
    full_response = ""
    async for chunk in rag_chain.astream({
        "history": history_str, 
        "context": code_units, 
        "question": rephrased_question
    }):
        if chunk.content:
            full_response += chunk.content
            await streaming_msg.stream_token(chunk.content)
    await streaming_msg.update()
    state["messages"].append(AIMessage(content=full_response))
    return state


def generate_answer(state: AgentState):
    history = state.get("conversation_history", [])
    code_units = state["code"]
    rephrased_question = state["rephrased_question"]
    history_str = ""
    for msg in history[-6:]: 
        if isinstance(msg, HumanMessage):
            history_str += f"**User:** {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            history_str += f"**GitSpeak:** {msg.content[:200]}...\n\n"
    
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
    error_message = """## Unable to Find Relevant Information

I apologize, but I couldn't find sufficient relevant information in the repository to answer your question comprehensively.

### Suggestions:
- Try rephrasing your question with more specific terms
- Ask about specific files, functions, or components
- Check if the information you're looking for might be in documentation files

**How can I help you explore this codebase differently?**"""
    
    state["messages"].append(AIMessage(content=error_message))
    return state


def off_topic(state: AgentState):
    """Handle off-topic questions"""
    off_topic_message = """## Off-Topic Question

I'm specialized in analyzing code repositories and helping with programming-related questions.

### I can help you with:
- **Code analysis** and explanation
- **Function and class** descriptions  
- **File structure** and organization
- **Documentation** review (README, comments)
- **Dependencies** and configuration
- **Best practices** and code review

**Please ask me something about this repository or programming in general!**"""
    
    state["messages"].append(AIMessage(content=off_topic_message))
    return state
def create_workflow():
    """Create and compile the workflow with SQLite checkpointer"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("batch_relevant_code_chunk", batch_relevant_code_chunk)
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
    workflow.add_edge("retriever", "batch_relevant_code_chunk")
    workflow.add_conditional_edges("batch_relevant_code_chunk", router, {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer"
    })
    workflow.add_edge("refine_question", "retriever")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("cannot_answer", END)
    workflow.set_entry_point("question_rewriter")
    
    # Initialize SQLite checkpointer properly with direct connection
    try:
        import sqlite3
        
        # Ensure database directory exists
        checkpoint_db_path = Path(CHECKPOINT_DB)
        checkpoint_db_path.parent.mkdir(exist_ok=True)
        
        # Create SQLite connection with proper threading support
        sqlite_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        
        # Use direct SqliteSaver constructor instead of context manager
        checkpointer = SqliteSaver(sqlite_conn)
        print(f"SQLite checkpointer initialized at: {CHECKPOINT_DB}")
        
    except Exception as e:
        print(f"Error initializing SQLite checkpointer: {e}")
        raise e
    
    return workflow.compile(checkpointer=checkpointer)

workflow_app = None

def get_workflow():
    """Get or create workflow instance"""
    global workflow_app
    if workflow_app is None:
        workflow_app = create_workflow()
    return workflow_app


@cl.on_chat_start
async def start():
    """Initialize chat session with persistent repository management"""
    existing_repos = find_existing_repositories()
    
    if existing_repos:
        repo_options = []
        for i, repo in enumerate(existing_repos, 1):
            repo_options.append(f"{i}. **{repo['repo_name']}** - `{repo['repo_url']}`")
        
        repo_list = "\n".join(repo_options)
        
        welcome_msg = f"""# GitSpeak - Repository Analysis

## Available Repositories

{repo_list}


"""

        await cl.Message(content=welcome_msg, author="GitSpeak").send()
        
        # Get user selection
        while True:
            user_input = await cl.AskUserMessage(
                content="**Your choice (number or GitHub URL):**",
                author="GitSpeak",
                timeout=300
            ).send()
            
            if not user_input:
                continue
                
            choice = user_input.get("output", "").strip()
            
            # Check if it's a number (existing repo selection)
            if choice.isdigit():
                repo_index = int(choice) - 1
                if 0 <= repo_index < len(existing_repos):
                    selected_repo = existing_repos[repo_index]
                    repo_hash = selected_repo['repo_hash']
                    
                    # Load repository
                    retriever, repo_name, _ = load_retriever_data(repo_hash)
                    if retriever:
                        await setup_repository_session(retriever, repo_name, selected_repo['repo_url'], repo_hash)
                        return
                else:
                    await cl.Message(
                        content="Invalid selection. Please choose a valid repository number.",
                        author="GitSpeak"
                    ).send()
                    continue
            
            # Check if it's a GitHub URL
            elif "github.com" in choice:
                await setup_new_repository(choice)
                return
                
            else:
                await cl.Message(
                    content="Please enter a valid repository number or GitHub URL.",
                    author="GitSpeak"
                ).send()
                continue
    else:
        welcome_msg = """# GitSpeak

Enhanced AI assistant for GitHub repository analysis.
"""

        await cl.Message(content=welcome_msg, author="GitSpeak").send()
        
        # Get repository URL
        while True:
            user_input = await cl.AskUserMessage(
                content="**Paste the GitHub Repository URL to start conversation:**",
                author="GitSpeak",
                timeout=300
            ).send()
            
            if user_input and "github.com" in user_input.get("output", ""):
                repo_url = user_input.get("output", "").strip()
                await setup_new_repository(repo_url)
                return
            else:
                await cl.Message(
                    content="Please provide a valid GitHub repository URL.",
                    author="GitSpeak"
                ).send()


async def setup_new_repository(repo_url: str):
    """Setup a new repository for analysis"""
    try:
        retriever, repo_name, repo_hash = await build_retriever_from_repo(repo_url)
        await setup_repository_session(retriever, repo_name, repo_url, repo_hash)
    except Exception as e:
        await cl.Message(
            content=f"""## Setup Error

**Failed to setup repository:** {str(e)}

### Troubleshooting:
- Ensure the repository URL is correct and public
- Check your internet connection
- Verify the repository exists and is accessible

Please try with a different repository URL.""",
            author="GitSpeak"
        ).send()


async def setup_repository_session(retriever, repo_name: str, repo_url: str, repo_hash: str):
    """Setup the chat session for a repository"""
    
    # Store in session
    cl.user_session.set("repo_name", repo_name)
    cl.user_session.set("repo_url", repo_url)
    cl.user_session.set("repo_hash", repo_hash)
    
    # Load existing conversation history
    session_data = load_session_data(repo_hash)
    conversation_history = session_data.get("conversation_history", [])
    cl.user_session.set("conversation_history", conversation_history)
    
    # Show repository info and status
    history_info = ""
    if conversation_history:
        history_info = f"\n\n**{len(conversation_history)} previous messages restored**"

    success_msg = f"""#### {repo_name} Ready for Analysis!"""

    await cl.Message(content=success_msg, author="GitSpeak").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages with LangGraph workflow and streaming"""
    
    # Check if repository is loaded
    repo_name = cl.user_session.get("repo_name")
    repo_hash = cl.user_session.get("repo_hash")
    
    if not repo_name or not repo_hash:
        await cl.Message(
            content="""## Repository Not Loaded

Please start a new chat session or select a repository first.

Use the **"New Chat"** button to begin!""",
            author="GitSpeak"
        ).send()
        return
    
    if repo_hash not in GLOBAL_RETRIEVERS:
        retriever, _, _ = load_retriever_data(repo_hash)
        if not retriever:
            await cl.Message(
                content="## Error: Repository data not found. Please start a new chat session.",
                author="GitSpeak"
            ).send()
            return
        
        # Cache the retriever
        GLOBAL_RETRIEVERS[repo_hash] = {
            'retriever': retriever,
            'repo_name': repo_name,
            'repo_url': cl.user_session.get("repo_url", "")
        }
    
    # Get conversation history
    conversation_history = cl.user_session.get("conversation_history", [])
    
    # Show typing indicator with streaming info
    async with cl.Step(name="Processing with LangGraph Streaming", type="run") as step:
        step.output = "Running question through LangGraph workflow with streaming enabled..."
        
        # Get workflow instance
        workflow = get_workflow()
        
        # Prepare initial state for LangGraph
        initial_state = {
            "messages": [],
            "on_topic": "",
            "rephrased_question": "",
            "proceed_to_generate": "",
            "rephrase_count": 0,
            "question": HumanMessage(content=message.content),
            "code": [],
            "repo_hash": repo_hash,
            "conversation_history": conversation_history
        }
        
        thread_config = {
            "configurable": {
                "thread_id": f"chat-{repo_hash}",
                "checkpoint_ns": "" 
            }
        }
        
        step.output = "📡 Executing LangGraph workflow with streaming and SQLite persistence..."
        
        # Check if we need to handle the generate_answer node specially for streaming
        try:
            # Process through workflow until we reach generate_answer
            current_state = initial_state
            should_stream = False
            
            # Run workflow step by step to detect when we reach generate_answer
            for step_result in workflow.stream(current_state, config=thread_config):
                node_name = list(step_result.keys())[0]
                current_state = step_result[node_name]
                
                if node_name == "generate_answer":
                    should_stream = True
                    break
            
            if should_stream:
                step.output = "Generating streaming response..."
                
                # Handle streaming response
                history = current_state.get("conversation_history", [])
                code_units = current_state["code"]
                rephrased_question = current_state["rephrased_question"]
                history_str = ""
                for msg in history[-6:]:
                    if isinstance(msg, HumanMessage):
                        history_str += f"**User:** {msg.content}\n\n"
                    elif isinstance(msg, AIMessage):
                        history_str += f"**GitSpeak:** {msg.content[:200]}...\n\n"
                
                step.output = "Streaming AI response in real-time..."
                
                # Create a new streaming message
                streaming_msg = cl.Message(content="", author="GitSpeak")
                await streaming_msg.send()
                
                # Stream the response
                full_response = ""
                async for chunk in rag_chain.astream({
                    "history": history_str, 
                    "context": code_units, 
                    "question": rephrased_question
                }):
                    if chunk.content:
                        full_response += chunk.content
                        await streaming_msg.stream_token(chunk.content)
                
                # Finalize the streaming message
                await streaming_msg.update()
                
                response = full_response
                
            else:
                result = workflow.invoke(initial_state, config=thread_config)
                
                if result.get("messages") and len(result["messages"]) > 0:
                    final_message = result["messages"][-1]
                    if isinstance(final_message, AIMessage):
                        response = final_message.content
                    else:
                        response = str(final_message)
                        
                    # Send non-streaming response
                    await cl.Message(
                        content=response,
                        author="GitSpeak"
                    ).send()
                else:
                    response = """## Processing Complete

I've analyzed your question through the LangGraph workflow, but couldn't generate a specific response. 

**Please try:**
- Being more specific about what you'd like to know
- Asking about particular files, functions, or components
- Rephrasing your question with different terms

**What would you like to explore in this repository?**"""
                    
                    await cl.Message(
                        content=response,
                        author="GitSpeak"
                    ).send()
            
            step.output = "LangGraph workflow with streaming completed successfully!"
            
        except Exception as workflow_error:
            print(f"Workflow error: {workflow_error}")
            step.output = f"Workflow error: {str(workflow_error)}"
            error_response = f"""## Workflow Processing Error

I encountered an error while processing your question through the LangGraph workflow:

```
{str(workflow_error)}
```

This might be due to:
- **Repository data issues** - Try reloading the repository
- **Workflow configuration** - The LangGraph setup may need adjustment
- **SQLite persistence issues** - Database connection problems
- **Streaming connection** - Network or API issues

**Please try:**
- Asking a simpler, more specific question
- Restarting the chat session
- Using a different repository

**Example questions that work well:**
- "What does the main.py file do?"
- "Explain the key functions in this code"
- "What's in the README file?"
"""
            
            # Send error as streaming message
            error_msg = cl.Message(content="", author="GitSpeak")
            await error_msg.send()
            for char in error_response:
                await error_msg.stream_token(char)
                await asyncio.sleep(0.02)  
            await error_msg.update()
            response = error_response
    
    # Update conversation history
    conversation_history.extend([
        HumanMessage(content=message.content),
        AIMessage(content=response)
    ])
    cl.user_session.set("conversation_history", conversation_history)
    
    # Save session data with persistent memory
    if repo_hash:
        session_data = {
            "repo_name": repo_name,
            "repo_url": cl.user_session.get("repo_url"),
            "conversation_history": conversation_history,
            "last_workflow_state": {
                "thread_id": f"chat-{repo_hash}",
                "timestamp": datetime.now().isoformat(),
                "streaming_enabled": True
            }
        }
        save_session_data(repo_hash, session_data)


# Enhanced streaming utilities
async def stream_message_gradually(message: str, author: str = "GitSpeak", delay: float = 0.02):
    """Stream a message character by character with customizable delay"""
    streaming_msg = cl.Message(content="", author=author)
    await streaming_msg.send()
    
    for char in message:
        await streaming_msg.stream_token(char)
        await asyncio.sleep(delay)
    
    await streaming_msg.update()
    return streaming_msg


async def stream_code_explanation(code_blocks: List[str], explanations: List[str], author: str = "GitSpeak"):
    """Stream code blocks with explanations in a structured way"""
    streaming_msg = cl.Message(content="", author=author)
    await streaming_msg.send()
    
    for i, (code, explanation) in enumerate(zip(code_blocks, explanations)):
        # Stream section header
        header = f"\n## Code Block {i+1}\n\n"
        for char in header:
            await streaming_msg.stream_token(char)
            await asyncio.sleep(0.01)
        
        # Stream code block
        code_section = f"```python\n{code}\n```\n\n"
        for char in code_section:
            await streaming_msg.stream_token(char)
            await asyncio.sleep(0.005)
        
        # Stream explanation
        for char in explanation:
            await streaming_msg.stream_token(char)
            await asyncio.sleep(0.02)
        
        await streaming_msg.stream_token("\n\n")
    
    await streaming_msg.update()
    return streaming_msg

def get_workflow_state(repo_hash: str) -> dict:
    """Get the current workflow state for a repository from SQLite"""
    try:
        workflow = get_workflow()
        thread_config = {
            "configurable": {
                "thread_id": f"chat-{repo_hash}",
                "checkpoint_ns": ""
            }
        }
        state = workflow.get_state(thread_config)
        return state.values if state else {}
    except Exception as e:
        print(f"Error getting workflow state: {e}")
        return {}


def reset_workflow_state(repo_hash: str):
    """Reset the workflow state for a repository in SQLite"""
    try:
        workflow = get_workflow()
        thread_config = {
            "configurable": {
                "thread_id": f"chat-{repo_hash}",
                "checkpoint_ns": ""
            }
        }
        # Clear the state by updating with empty state
        workflow.update_state(thread_config, {})
        print(f"Reset workflow state for {repo_hash}")
    except Exception as e:
        print(f"Error resetting workflow state: {e}")


def list_workflow_threads():
    """List all workflow threads stored in SQLite"""
    try:
        workflow = get_workflow()
        # Get checkpointer instance
        checkpointer = workflow.checkpointer
        
        # List all threads/configurations
        configs = []
        for config_data in checkpointer.list({}):
            configs.append(config_data.config)
        
        return configs
    except Exception as e:
        print(f"Error listing workflow threads: {e}")
        return []


def get_thread_history(repo_hash: str):
    """Get complete thread history from SQLite"""
    try:
        workflow = get_workflow()
        thread_config = {
            "configurable": {
                "thread_id": f"chat-{repo_hash}",
                "checkpoint_ns": ""
            }
        }
        
        # Get state history
        history = []
        for state_snapshot in workflow.get_state_history(thread_config):
            history.append({
                'config': state_snapshot.config,
                'values': state_snapshot.values,
                'next': state_snapshot.next,
                'metadata': state_snapshot.metadata,
                'created_at': state_snapshot.created_at
            })
        
        return history
    except Exception as e:
        print(f"Error getting thread history: {e}")
        return []


# Enhanced chat profile with streaming information
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GitSpeak-Streaming",
            markdown_description="""**AI-Powered Repository Analysis **""",
            icon="https://github.com/favicon.ico",
        ),
    ]

def cleanup_old_checkpoints(days_old: int = 30):
    """Clean up old checkpoint data from SQLite database"""
    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)
        workflow = get_workflow()
        checkpointer = workflow.checkpointer
        print(f"Cleanup functionality would remove checkpoints older than {cutoff_date}")
        print("SQLite cleanup requires direct database operations")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")


def get_sqlite_database_stats():
    """Get statistics about the SQLite database"""
    try:
        import sqlite3
        
        if not Path(CHECKPOINT_DB).exists():
            return {"status": "Database not found"}
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        stats = {
            "database_path": CHECKPOINT_DB,
            "database_size_bytes": Path(CHECKPOINT_DB).stat().st_size,
            "tables": [table[0] for table in tables],
            "total_tables": len(tables),
            "streaming_enabled": True
        }
        for table_name in stats["tables"]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                stats[f"{table_name}_rows"] = count
            except sqlite3.Error as e:
                stats[f"{table_name}_error"] = str(e)
        
        conn.close()
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def backup_sqlite_database(backup_path: str = None):
    """Create a backup of the SQLite database"""
    try:
        import shutil
        import sqlite3
        
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(PERSISTENT_DIR / f"checkpoint_backup_{timestamp}.sqlite")
        
        if Path(CHECKPOINT_DB).exists():
            # Use proper SQLite backup method
            source_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
            backup_conn = sqlite3.connect(backup_path, check_same_thread=False)
            
            # Perform the backup
            source_conn.backup(backup_conn)
            
            # Close connections
            backup_conn.close()
            source_conn.close()
            
            print(f"Database backed up to: {backup_path}")
            return backup_path
        else:
            print("No database file found to backup")
            return None
            
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None
def cleanup_old_sessions(days_old: int = 30):
    """Clean up old session data and checkpoints"""
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days_old)
    for session_file in PERSISTENT_DIR.glob("session_*.json"):
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            last_updated = datetime.fromisoformat(data.get('last_updated', '1970-01-01'))
            if last_updated < cutoff_date:
                session_file.unlink()
                print(f"Removed old session: {session_file.name}")
        except Exception as e:
            print(f"Error cleaning session {session_file}: {e}")
    cleanup_old_checkpoints(days_old)


def list_active_repositories():
    """List all active repositories with their metadata"""
    repos = find_existing_repositories()
    for repo in repos:
        print(f"Repository: {repo['repo_name']}")
        print(f"  URL: {repo['repo_url']}")
        print(f"  Hash: {repo['repo_hash']}")
        print(f"  Created: {repo['created_at']}")
        thread_history = get_thread_history(repo['repo_hash'])
        print(f"  SQLite Checkpoints: {len(thread_history)}")
        print(f"  Streaming Enabled: Yes")
        print("-" * 50)


def get_repository_stats(repo_hash: str):
    """Get statistics for a specific repository including streaming data"""
    try:
        session_data = load_session_data(repo_hash)
        sqlite_stats = get_sqlite_database_stats()
        thread_history = get_thread_history(repo_hash)
        
        if session_data:
            history_count = len(session_data.get('conversation_history', []))
            last_updated = session_data.get('last_updated', 'Unknown')
            streaming_enabled = session_data.get('last_workflow_state', {}).get('streaming_enabled', False)
            
            print(f"Repository Statistics for {repo_hash}:")
            print(f"  Messages in history: {history_count}")
            print(f"  Last updated: {last_updated}")
            print(f"  SQLite checkpoints: {len(thread_history)}")
            print(f"  Streaming enabled: {streaming_enabled}")
            retriever_dir = PERSISTENT_DIR / f"retriever_{repo_hash}"
            if retriever_dir.exists():
                metadata_file = retriever_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"  Repository: {metadata['repo_name']}")
                    print(f"  URL: {metadata['repo_url']}")
                    print(f"  Vector store created: {metadata['created_at']}")
            
            print(f"\nSQLite Database Stats:")
            for key, value in sqlite_stats.items():
                print(f"  {key}: {value}")
            
            return {
                'history_count': history_count,
                'last_updated': last_updated,
                'sqlite_checkpoints': len(thread_history),
                'has_vector_store': (retriever_dir / "faiss_store").exists(),
                'sqlite_stats': sqlite_stats,
                'streaming_enabled': streaming_enabled
            }
    except Exception as e:
        print(f"Error getting stats for {repo_hash}: {e}")
        return None



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GitSpeak with LangGraph, SQLite, and Streaming')
    parser.add_argument('--cleanup', type=int, help='Clean up sessions older than N days')
    parser.add_argument('--list-repos', action='store_true', help='List all repositories')
    parser.add_argument('--stats', type=str, help='Get stats for repository hash')
    parser.add_argument('--db-stats', action='store_true', help='Show SQLite database statistics')
    parser.add_argument('--backup-db', type=str, nargs='?', const='', help='Backup SQLite database')
    parser.add_argument('--list-threads', action='store_true', help='List all workflow threads')
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_old_sessions(args.cleanup)
    elif args.list_repos:
        list_active_repositories()
    elif args.stats:
        get_repository_stats(args.stats)
    elif args.db_stats:
        stats = get_sqlite_database_stats()
        print("SQLite Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    elif args.backup_db is not None:
        backup_path = args.backup_db if args.backup_db else None
        backup_sqlite_database(backup_path)
    elif args.list_threads:
        threads = list_workflow_threads()
        print(f"Active Workflow Threads: {len(threads)}")
        for thread in threads:
            print(f"  Thread: {thread}")
    else:
        print(f"Starting GitSpeak with SQLite persistence and streaming")
        print(f"Persistent directory: {PERSISTENT_DIR}")
        print(f"SQLite database: {CHECKPOINT_DB}")
        print(f"Streaming enabled: True")
        
        # Get port from environment variable (Render sets this automatically)
        port = 8000
        cl.run(
            debug=False, 
            watch=False, 
            port=port, 
            host='0.0.0.0',
            headless=True 
        )