import genaihub_client
genaihub_client.set_environment_variables()

from gen_ai_hub.proxy.langchain.init_models import init_llm
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dataset_loader import MarkdownDatasetLoader
import requests

llm = init_llm('gpt-4o-mini', max_tokens=300)

class State(TypedDict):
    filename: str
    text: str
    images: list
    urls: list
    quality_report: Annotated[list, add_messages]
    sap_products: Annotated[list, add_messages]
    url_report: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def analyze_quality(state: State):
    prompt = (
        "Analyze the following markdown content for quality issues (structure, spelling, formatting, clarity). "
        "Return a short report:\n\n"
        f"{state['text']}"
    )
    result = llm.invoke([{"role": "user", "content": prompt}])
    return {"quality_report": [result]}

def extract_sap_products(state: State):
    prompt = (
        "Extract all SAP product names mentioned in the following markdown content. "
        "Return as a comma-separated list:\n\n"
        f"{state['text']}"
    )
    result = llm.invoke([{"role": "user", "content": prompt}])
    return {"sap_products": [result]}

def check_urls(state: State):
    report = []
    for url in state["urls"]:
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            status = response.status_code
            if status == 403:
                report.append(f"{url}: Authorization missing (HTTP {status})")
            elif status >= 400:
                report.append(f"{url}: Error (HTTP {status})")
            else:
                report.append(f"{url}: OK (HTTP {status})")
        except Exception as e:
            report.append(f"{url}: Error ({e})")
    return {"url_report": report}

def get_product_context(product_name):
    result = None # add information from call_grounding_api.py here
    return result

# Add nodes to the graph
graph_builder.add_node("analyze_quality", analyze_quality)
graph_builder.add_node("extract_sap_products", extract_sap_products)
graph_builder.add_node("check_urls", check_urls)

# Define the flow: START -> analyze_quality -> extract_sap_products -> check_urls -> END
graph_builder.add_edge(START, "analyze_quality")
graph_builder.add_edge("analyze_quality", "extract_sap_products")
graph_builder.add_edge("extract_sap_products", "check_urls")
graph_builder.add_edge("check_urls", END)

graph = graph_builder.compile()

def analyze_markdown_files(subfolder):
    loader = MarkdownDatasetLoader()
    loader.load_from_subfolder(subfolder)
    for filename, filedata in loader.data.items():
        print(f"\n--- Analyzing: {filename} ---")
        state = {
            "filename": filename,
            "text": filedata["text"],
            "images": filedata["images"],
            "urls": filedata["urls"],
            "quality_report": [],
            "sap_products": [],
            "url_report": []
        }
        for event in graph.stream(state):
            for value in event.values():
                if "quality_report" in value:
                    print("Quality Report:", value["quality_report"][-1].content)
                if "sap_products" in value:
                    print("SAP Products:", value["sap_products"][-1].content)
                if "url_report" in value:
                    print("URL Report:")
                    for line in value["url_report"]:
                        print("  ", line)

# add a node that checks sap product names in a datastore with all product names and corresponding embeddings, first check keyword based then embedding if naming is wrong


# remember playwright tool for second phase of the project