import asyncio
from pathlib import Path
import os
import neo4j
from neo4j_graphrag.embeddings import CohereEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import AnthropicLLM
from typing import List

from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(override=True)

# Neo4j db infos
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Provide default if not set
AUTH = (
    os.getenv("NEO4J_USERNAME"),
    os.getenv("NEO4J_PASSWORD"),
)  # Should be a tuple (user, password) or Neo4j auth object
DATABASE = os.getenv("NEO4J_DB_NAME", "neo4j")  # Default database name

# LLM Configuration (Example using VertexAI, adjust as needed)
# Ensure VERTEXAI_PROJECT_ID and potentially GOOGLE_APPLICATION_CREDENTIALS are set in env
CLAUDE_API_KEY = os.getenv("CLAUDE_API")
# Ensure COHERE_API_KEY is set in your environment for CohereEmbeddings
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Define the path to your input file or folder
# Example: process a single file
# INPUT_PATH_STR = "data/Harry Potter and the Chamber of Secrets Summary.pdf"
# Example: process all PDFs in a folder
INPUT_PATH_STR = "data/Automatic Knowledge Graph Construction Survey.pdf"  # Assuming 'data' folder exists relative to parent dir

# Define Entities and Relations for Knowledge Graph Extraction
ENTITIES = [
    "KnowledgeGraph",  # The central structured knowledge representation [cite: 7, 32, 100]
    "Data",  # Input sources like text, tables, web pages [cite: 2, 26, 31, 36]
    "Entity",  # Nodes in the KG (concepts, attributes, instances) [cite: 8, 33, 99, 101, 210]
    "Relation",  # Edges connecting entities/concepts in the KG [cite: 34, 95, 100]
    "Triple",  # Basic fact unit (head, relation, tail) [cite: 33, 101]
    "Method",  # Algorithms or models used (e.g., Deep Learning, GCN, SVM) [cite: 6, 50, 51, 76]
    "Task",  # Specific processes (e.g., NER, Relation Extraction, KGC, Entity Linking) [cite: 7, 9, 50, 207, 565]
    "ProcessStage",  # High-level stages like Knowledge Acquisition, Refinement, Evolution [cite: 7, 26, 28]
    "Condition",  # Contextual factors like time or prerequisites [cite: 10, 58, 722, 723]
    "Resource",  # Datasets, tools, projects (e.g., Freebase, spaCy, DBpedia) [cite: 12, 79, 106, 150]
    "Challenge",  # Issues addressed (e.g., Noisy Data, Low-Resource Data, Long Context) [cite: 55, 68, 464, 801]
]
RELATIONS = [
    "HAS_INPUT",  # Connects a Task/Method to the Data it uses [cite: 26, 31, 206]
    "HAS_OUTPUT",  # Connects a Task/Method to what it produces [cite: 26, 170]
    "CONSISTS_OF",  # Connects a KG to its components (Entity, Relation, Triple) [cite: 32, 34, 100]
    "APPLIES_TO",  # Connects a Method/Task to the element it acts upon [cite: 7, 9, 51, 564]
    "PART_OF",  # Connects a sub-task/component to a larger Task/ProcessStage [cite: 7, 207, 210]
    "USES",  # Connects a Task/Method to a Resource it utilizes [cite: 45, 114, 150, 398]
    "ADDRESSES",  # Connects a Method/Task to a Challenge it tackles [cite: 55, 464, 468]
    "DEFINED_BY",  # Connects an element (e.g., Triple) to its defining features (e.g., Condition) [cite: 723]
    "LINKS",  # Connects EntityLinking task to Entities and KGs [cite: 213, 277]
    "REFINES",  # Connects Knowledge Refinement to a KG [cite: 9, 27, 51, 564]
    "EVOLVES_WITH",  # Connects a KG to Conditions affecting its state [cite: 10, 29, 58, 721, 724]
    "INCLUDES_TASK",  # Connects a ProcessStage to the Tasks within it [cite: 7, 207, 565]
    "EXTRACTS",  # Connects a Task/Method to the Entity/Relation it identifies [cite: 2, 8, 43, 157]
    "COMPLETES",  # Connects KGC Task/Method to Triples/KG [cite: 9, 52, 567]
    "FUSES",  # Connects Knowledge Fusion Task/Method to KGs/Entities [cite: 9, 52, 566, 646]
    "RESOLVES",  # Connects Coreference Resolution Task to Mentions/Entities [cite: 9, 43, 208, 333]
    "DISCOVERS",  # Connects Entity Discovery Task to Entities [cite: 9, 43, 209]
]

POTENTIAL_SCHEMA = [
    ("Task", "HAS_INPUT", "Data"),
    ("Method", "HAS_OUTPUT", "KnowledgeGraph"),
    ("KnowledgeGraph", "CONSISTS_OF", "Entity"),
    ("KnowledgeGraph", "CONSISTS_OF", "Relation"),
    ("Method", "APPLIES_TO", "KnowledgeGraph"),
    ("Task", "APPLIES_TO", "Data"),
    ("Task", "PART_OF", "ProcessStage"),
    ("KnowledgeAcquisition", "INCLUDES_TASK", "EntityDiscovery"),
    ("KnowledgeAcquisition", "INCLUDES_TASK", "RelationExtraction"),
    ("EntityDiscovery", "INCLUDES_TASK", "NamedEntityRecognition"),
    ("EntityDiscovery", "INCLUDES_TASK", "EntityTyping"),
    ("EntityDiscovery", "INCLUDES_TASK", "EntityLinking"),
    ("Method", "USES", "Resource"),
    ("Task", "ADDRESSES", "Challenge"),
    (
        "Triple",
        "DEFINED_BY",
        "Condition",
    ),  # e.g., (ConditionalTriple, DEFINED_BY, TimeCondition)
    ("EntityLinking", "LINKS", "Entity"),
    ("KnowledgeRefinement", "REFINES", "KnowledgeGraph"),
    ("KnowledgeGraph", "EVOLVES_WITH", "Condition"),
    ("RelationExtraction", "EXTRACTS", "Relation"),
    ("NamedEntityRecognition", "EXTRACTS", "Entity"),
    ("KnowledgeGraphCompletion", "COMPLETES", "Triple"),
    ("KnowledgeFusion", "FUSES", "Entity"),
    ("CoreferenceResolution", "RESOLVES", "Entity"),
]
# --- End Configuration ---


# --- Helper Functions ---
def get_pdf_files(input_path: Path) -> List[Path]:
    """
    Identifies PDF files based on the input path.

    Args:
        input_path: A Path object pointing to a file or directory.

    Returns:
        A list of Path objects, each pointing to a PDF file to be processed.
        Returns an empty list if no valid PDFs are found.
    """
    pdf_files_to_process = []
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return []

    if input_path.is_dir():
        print(f"Input path is a directory: {input_path}. Searching for PDF files...")
        pdf_files_to_process = list(input_path.glob("*.pdf"))
        if not pdf_files_to_process:
            print(f"No PDF files found in directory: {input_path}")
        else:
            print(f"Found {len(pdf_files_to_process)} PDF files.")
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        print(f"Input path is a single PDF file: {input_path}")
        pdf_files_to_process.append(input_path)
    else:
        print(f"Error: Input path is not a directory or a valid PDF file: {input_path}")

    return pdf_files_to_process


# --- Core Pipeline Logic ---
async def run_pipeline_for_file(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    embedder: CohereEmbeddings,
    file_path: Path,
) -> PipelineResult | None:
    """
    Defines and runs the SimpleKGPipeline for a single file.

    Args:
        neo4j_driver: Initialized Neo4j driver instance.
        llm: Initialized LLM interface instance.
        embedder: Initialized Embeddings instance.
        file_path: Path object for the PDF file to process.

    Returns:
        PipelineResult object if successful, None otherwise.
    """
    print(f"\n--- Processing file: {file_path.name} ---")
    try:
        # Create an instance of the SimpleKGPipeline for each file
        # This ensures clean state if needed, though builder might be reusable
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=neo4j_driver,
            embedder=embedder,
            entities=ENTITIES,
            relations=RELATIONS,
            potential_schema=POTENTIAL_SCHEMA,
            neo4j_database=DATABASE,
            # Add other relevant parameters like chunk_size if needed
            # chunk_size=1024,
            # chunk_overlap=200,
        )
        result = await kg_builder.run_async(file_path=str(file_path))
        print(f"Successfully processed: {file_path.name}")
        # You might want to inspect the result here, e.g., result.graph_summary
        return result
    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        # Depending on the error, you might want to log more details
        # import traceback
        # traceback.print_exc()
        return None


# --- Main Execution Function ---
async def main(file_paths: List[Path]) -> List[PipelineResult | None]:
    """
    Sets up resources and orchestrates the processing of multiple PDF files.

    Args:
        file_paths: A list of Path objects for the PDF files to process.

    Returns:
        A list containing PipelineResult objects for successfully processed files
        and None for files that encountered errors.
    """
    if not file_paths:
        print("No PDF files provided to process.")
        return []

    # --- Resource Initialization ---
    # Initialize LLM (Ensure API keys/credentials are set in environment)
    # NOTE: Using gpt-4o with VertexAI might require specific setup or proxy.
    #       Consider using a native Vertex AI model like 'gemini-1.5-pro-preview-0409'
    #       or ensure your VertexAI client is configured for OpenAI models if applicable.
    #       Using a placeholder model name here for demonstration.
    #       Replace "your-actual-vertex-model" with a valid model like "gemini-pro"
    llm = AnthropicLLM(
        # model_name="gemini-1.5-pro-preview-0409", # Example Gemini model
        model_name="claude-3-7-sonnet-20250219",  # More standard Gemini model
        api_key=CLAUDE_API_KEY,
        model_params={
            "max_output_tokens": 2048,  # Adjusted parameter name for model
            # "response_format": {"type": "json_object"}, # JSON format may not be directly supported/needed depending on model/task
            "temperature": 0.7,  # Example parameter
        },
    )
    print("LLM Initialized.")

    # Initialize Embedder (Ensure COHERE_API_KEY is set)
    if not COHERE_API_KEY:
        print("Error: COHERE_API_KEY environment variable not set.")
        # Close LLM client before exiting if necessary
        await llm.async_client.close()
        return [None] * len(file_paths)  # Return None for all expected results

    try:
        embedder = CohereEmbeddings(api_key=COHERE_API_KEY)
        print("Embedder Initialized.")
    except Exception as e:
        print(f"Error initializing Cohere Embeddings: {e}")
        await llm.async_client.close()
        return [None] * len(file_paths)

    # Initialize Neo4j Driver
    if not URI or not AUTH:
        print("Error: NEO4J_URI or NEO4J_AUTH environment variables not set.")
        await llm.async_client.close()
        return [None] * len(file_paths)

    results = []
    try:
        # Use try-with-resources for the driver
        with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
            print(f"Neo4j Driver Connected to {URI}, Database: {DATABASE}")
            driver.verify_connectivity()  # Check connection
            print("Neo4j Connection Verified.")

            # Process each file sequentially
            for file_path in file_paths:
                res = await run_pipeline_for_file(driver, llm, embedder, file_path)
                results.append(res)

    except neo4j.exceptions.AuthError as e:
        print(f"Neo4j Authentication Error: {e}. Check NEO4J_AUTH.")
    except neo4j.exceptions.ServiceUnavailable as e:
        print(
            f"Neo4j Connection Error: {e}. Check NEO4J_URI and if the database is running."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred during Neo4j connection or processing: {e}"
        )
        # Ensure results list matches expected length if error happens mid-processing
        results.extend([None] * (len(file_paths) - len(results)))
    finally:
        # --- Resource Cleanup ---
        # Close the LLM client after all processing is done or if an error occurs
        await llm.async_client.close()
        print("LLM Client Closed.")

    return results


# --- Script Entry Point ---
if __name__ == "__main__":
    print("--- Starting Knowledge Graph Ingestion Script ---")

    # Determine the absolute path for input based on the script's location
    script_dir = Path(__file__).parent
    input_path = (
        script_dir / INPUT_PATH_STR
    ).resolve()  # Use resolve for absolute path

    # Get the list of PDF files to process
    pdf_files = get_pdf_files(input_path)

    if pdf_files:
        print(f"\nStarting processing for {len(pdf_files)} file(s)...")
        # Run the main asynchronous function
        pipeline_results = asyncio.run(main(pdf_files))

        print("\n--- Processing Summary ---")
        successful_ingestions = 0
        failed_ingestions = 0
        for i, result in enumerate(pipeline_results):
            file_name = pdf_files[i].name
            if result is not None:
                # You can print more details from the result if needed
                # print(f"Result for {file_name}: {result}")
                print(f"✅ Successfully processed: {file_name}")
                successful_ingestions += 1
            else:
                print(f"❌ Failed to process: {file_name}")
                failed_ingestions += 1

        print(f"\nTotal Files Processed: {len(pdf_files)}")
        print(f"Successful: {successful_ingestions}")
        print(f"Failed: {failed_ingestions}")
    else:
        print("No valid PDF files found to process. Exiting.")

    print("\n--- Script Finished ---")
