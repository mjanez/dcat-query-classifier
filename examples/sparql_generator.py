"""
Example: SPARQL Query Generator using SPARQL Humanize

This example demonstrates how to use SPARQL Humanize to build
a natural language interface for SPARQL queries.
"""

import sys
from pathlib import Path

# Add parent directory to path to import sparql_humanize
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparql_humanize.predict import predict  # noqa: E402
from typing import Dict, Any  # noqa: E402
import re  # noqa: E402


# SPARQL query templates for each intent type
SPARQL_TEMPLATES = {
    "list_datasets": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?dataset ?title ?description
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dct:description ?description .
}}
LIMIT 100
""",
    
    "search_datasets": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?dataset ?title ?description
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dct:description ?description ;
             dcat:theme ?theme .
    FILTER(CONTAINS(LCASE(STR(?theme)), "{keyword}") || 
           CONTAINS(LCASE(?title), "{keyword}") ||
           CONTAINS(LCASE(?description), "{keyword}"))
}}
LIMIT 50
""",
    
    "get_distribution": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?dataset ?title ?distribution ?format
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dcat:distribution ?distribution .
    ?distribution dcat:mediaType ?format .
    FILTER(CONTAINS(LCASE(?format), "{format}"))
}}
LIMIT 50
""",
    
    "get_metadata": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?title ?description ?publisher ?issued ?modified
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dct:description ?description ;
             dct:publisher ?publisher ;
             dct:issued ?issued ;
             dct:modified ?modified .
    FILTER(CONTAINS(LCASE(?title), "{keyword}"))
}}
LIMIT 10
""",
    
    "filter_datasets": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?dataset ?title ?issued
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dct:issued ?issued .
    FILTER(?issued {operator} "{date}"^^xsd:date)
}}
ORDER BY DESC(?issued)
LIMIT 50
""",
    
    "get_statistics": """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT (COUNT(?dataset) as ?count) ?theme
WHERE {{
    ?dataset a dcat:Dataset ;
             dcat:theme ?theme .
}}
GROUP BY ?theme
ORDER BY DESC(?count)
"""
}


def extract_parameters(question: str, query_type: str) -> Dict[str, str]:
    """
    Extract parameters from the question based on query type.
    
    This is a simple implementation. In production, you'd use NER
    (Named Entity Recognition) or more sophisticated NLP.
    """
    params = {}
    question_lower = question.lower()
    
    if query_type == "search_datasets":
        # Extract theme/keyword
        keywords = ["health", "education", "environment", "transport", 
                   "economy", "climate", "salud", "educación"]
        for keyword in keywords:
            if keyword in question_lower:
                params["keyword"] = keyword
                break
        if "keyword" not in params:
            params["keyword"] = ""
    
    elif query_type == "get_distribution":
        # Extract format
        formats = {
            "csv": "csv", "json": "json", "xml": "xml",
            "excel": "xlsx", "rdf": "rdf"
        }
        for fmt_name, fmt_value in formats.items():
            if fmt_name in question_lower:
                params["format"] = fmt_value
                break
        if "format" not in params:
            params["format"] = ""
    
    elif query_type == "filter_datasets":
        # Extract date and operator
        # Simple date extraction (you'd use proper NER in production)
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            year = year_match.group(1)
            params["date"] = f"{year}-01-01"
        else:
            params["date"] = "2020-01-01"
        
        # Determine operator
        if any(word in question_lower for word in ["after", "since", "from", "después"]):
            params["operator"] = ">"
        elif any(word in question_lower for word in ["before", "until", "antes"]):
            params["operator"] = "<"
        else:
            params["operator"] = ">"
    
    elif query_type == "get_metadata":
        # Extract dataset name/keyword
        keywords = ["transportation", "health", "education", "climate", "economy"]
        for keyword in keywords:
            if keyword in question_lower:
                params["keyword"] = keyword
                break
        if "keyword" not in params:
            # Extract quoted text or last word
            params["keyword"] = ""
    
    return params


def generate_sparql(question: str) -> Dict[str, Any]:
    """
    Generate a SPARQL query from a natural language question.
    
    Args:
        question: Natural language question
    
    Returns:
        Dictionary with intent, confidence, SPARQL query, and explanation
    """
    # 1. Classify the intent
    result = predict(question)
    query_type = result["predicted_query_type"]
    confidence = result["confidence"]
    
    # 2. Get the appropriate template
    template = SPARQL_TEMPLATES.get(query_type, SPARQL_TEMPLATES["list_datasets"])
    
    # 3. Extract parameters from the question
    params = extract_parameters(question, query_type)
    
    # 4. Fill in the template
    try:
        sparql_query = template.format(**params)
    except KeyError:
        # If a parameter is missing, use the default template
        sparql_query = template
    
    # 5. Generate human-readable explanation
    explanations = {
        "list_datasets": "Listar todos los datasets disponibles",
        "search_datasets": f"Buscar datasets sobre '{params.get('keyword', 'tema')}'",
        "get_distribution": f"Obtener datasets en formato {params.get('format', 'específico')}",
        "get_metadata": f"Obtener metadatos del dataset '{params.get('keyword', 'especificado')}'",
        "filter_datasets": f"Filtrar datasets por fecha ({params.get('operator', '>')} {params.get('date', 'fecha')})",
        "get_statistics": "Obtener estadísticas de los datasets"
    }
    
    return {
        "intent": query_type,
        "confidence": confidence,
        "sparql": sparql_query,
        "explanation": explanations.get(query_type, "Consulta SPARQL generada"),
        "parameters": params
    }


def demo():
    """Run a demonstration of the SPARQL query generator."""
    
    print("=" * 80)
    print("SPARQL Humanize - Natural Language to SPARQL Demo")
    print("=" * 80)
    print()
    
    # Example questions
    questions = [
        "What datasets are available?",
        "Show me datasets about health",
        "I need datasets in CSV format",
        "Datasets created after 2020",
        "How many datasets do we have?",
        "Tell me about the education dataset"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: \"{question}\"")
        print("-" * 80)
        
        try:
            result = generate_sparql(question)
            
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Explanation: {result['explanation']}")
            
            if result['parameters']:
                print(f"   Parameters: {result['parameters']}")
            
            print("\n   Generated SPARQL Query:")
            print("   " + "\n   ".join(result['sparql'].strip().split('\n')))
            
        except Exception as e:
            print(f"   Error: {str(e)}")
        
        print()


if __name__ == "__main__":
    demo()
