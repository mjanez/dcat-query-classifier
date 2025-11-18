"""SPARQL Query Generator.

This module generates complete SPARQL queries based on the predicted intent
and extracted parameters from natural language questions.
"""

from typing import Dict, Any, Optional
import re
from .logger import get_logger

logger = get_logger(__name__)


class SPARQLGenerator:
    """Generate SPARQL queries from natural language intents."""

    # SPARQL prefixes commonly used in DCAT catalogs
    PREFIXES = """PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

"""

    def __init__(self):
        """Initialize the SPARQL generator."""
        self.templates = {
            "list_datasets": self._list_datasets,
            "search_datasets": self._search_datasets,
            "get_distribution": self._get_distribution,
            "get_metadata": self._get_metadata,
            "filter_datasets": self._filter_datasets,
            "get_statistics": self._get_statistics,
            "list_graphs": self._list_graphs,
            "list_services": self._list_services,
            "list_publishers": self._list_publishers,
            "list_properties": self._list_properties,
            "list_classes": self._list_classes,
            "list_publisher_properties": self._list_publisher_properties,
            "list_service_properties": self._list_service_properties,
            "rank_publishers": self._rank_publishers,
            "rank_publishers_services": self._rank_publishers_services,
            "rank_publishers_combined": self._rank_publishers_combined,
            "count_by_graph": self._count_by_graph,
            "filter_by_graph": self._filter_by_graph,
            "filter_publishers_language": self._filter_publishers_language,
        }

    def generate(
        self,
        query_type: str,
        question: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a SPARQL query based on intent and parameters.

        Args:
            query_type: The predicted query type/intent.
            question: Original natural language question.
            params: Optional parameters extracted from the question.

        Returns:
            Dictionary with:
                - sparql: The generated SPARQL query
                - description: Human-readable description
                - query_type: The intent type
        """
        if params is None:
            params = self._extract_parameters(question, query_type)

        if query_type not in self.templates:
            logger.warning(f"Unknown query type: {query_type}")
            return {
                "sparql": self._default_query(),
                "description": f"Query type '{query_type}' not yet implemented",
                "query_type": query_type,
                "params": params,
            }

        generator_func = self.templates[query_type]
        sparql_query = generator_func(params)

        return {
            "sparql": self.PREFIXES + sparql_query,
            "description": self._get_description(query_type, params),
            "query_type": query_type,
            "params": params,
        }

    def _extract_parameters(self, question: str, query_type: str) -> Dict[str, Any]:
        """Extract parameters from the natural language question.

        Args:
            question: Natural language question.
            query_type: Predicted query type.

        Returns:
            Dictionary of extracted parameters.
        """
        params = {}
        question_lower = question.lower()

        # Extract format/media type
        formats = ["csv", "json", "xml", "rdf", "excel", "turtle", "n-triples"]
        for fmt in formats:
            if fmt in question_lower:
                params["format"] = fmt
                break

        # Extract keywords/themes
        theme_keywords = ["health", "education", "transport", "economy", "climate", 
                         "salud", "educación", "transporte", "economía", "clima"]
        for keyword in theme_keywords:
            if keyword in question_lower:
                params["keyword"] = keyword
                break

        # Extract year/date
        year_match = re.search(r'\b(19|20)\d{2}\b', question)
        if year_match:
            params["year"] = year_match.group(0)

        # Extract limit
        if "top" in question_lower or "first" in question_lower:
            limit_match = re.search(r'\b(\d+)\b', question)
            if limit_match:
                params["limit"] = int(limit_match.group(0))
            else:
                params["limit"] = 10
        else:
            params["limit"] = 100

        # Extract language
        if "spanish" in question_lower or "español" in question_lower:
            params["language"] = "es"
        elif "english" in question_lower or "inglés" in question_lower:
            params["language"] = "en"

        return params

    def _get_description(self, query_type: str, params: Dict[str, Any]) -> str:
        """Get human-readable description of the query."""
        descriptions = {
            "list_datasets": "List all available datasets",
            "search_datasets": f"Search datasets about '{params.get('keyword', 'topic')}'",
            "get_distribution": f"Get datasets in {params.get('format', 'specified')} format",
            "get_metadata": "Get metadata for a dataset",
            "filter_datasets": f"Filter datasets by date (year: {params.get('year', 'specified')})",
            "get_statistics": "Get statistics about datasets",
            "list_graphs": "List all named graphs in the catalog",
            "list_services": "List all data services",
            "list_publishers": "List all dataset publishers",
            "list_properties": "List all dataset properties",
            "list_classes": "List all classes/types in the catalog",
            "list_publisher_properties": "List properties of publishers",
            "list_service_properties": "List properties of data services",
            "rank_publishers": "Rank publishers by number of datasets",
            "rank_publishers_services": "Rank publishers by number of services",
            "rank_publishers_combined": "Rank publishers by total resources",
            "count_by_graph": "Count resources by graph",
            "filter_by_graph": "Filter datasets by graph",
            "filter_publishers_language": f"Filter publishers by language ({params.get('language', 'specified')})",
        }
        return descriptions.get(query_type, f"Execute {query_type} query")

    # Template methods for each query type

    def _list_datasets(self, params: Dict[str, Any]) -> str:
        """Generate query to list all datasets."""
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?dataset ?title ?description
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title .
    OPTIONAL {{ ?dataset dct:description ?description . }}
}}
LIMIT {limit}"""

    def _search_datasets(self, params: Dict[str, Any]) -> str:
        """Generate query to search datasets by keyword."""
        keyword = params.get("keyword", "")
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?dataset ?title ?description
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title .
    OPTIONAL {{ ?dataset dct:description ?description . }}
    OPTIONAL {{ ?dataset dcat:keyword ?keyword . }}
    FILTER(
        CONTAINS(LCASE(?title), "{keyword}") ||
        CONTAINS(LCASE(?description), "{keyword}") ||
        CONTAINS(LCASE(?keyword), "{keyword}")
    )
}}
LIMIT {limit}"""

    def _get_distribution(self, params: Dict[str, Any]) -> str:
        """Generate query to filter by format/distribution."""
        format_type = params.get("format", "csv")
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?dataset ?title ?distribution ?format
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title ;
             dcat:distribution ?distribution .
    ?distribution dcat:mediaType ?format .
    FILTER(CONTAINS(LCASE(?format), "{format_type}"))
}}
LIMIT {limit}"""

    def _get_metadata(self, params: Dict[str, Any]) -> str:
        """Generate query to get dataset metadata."""
        return """SELECT ?dataset ?property ?value
WHERE {
    ?dataset a dcat:Dataset .
    ?dataset ?property ?value .
}
LIMIT 100"""

    def _filter_datasets(self, params: Dict[str, Any]) -> str:
        """Generate query to filter datasets by date."""
        year = params.get("year", "2020")
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?dataset ?title ?created ?modified
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:title ?title .
    OPTIONAL {{ ?dataset dct:created ?created . }}
    OPTIONAL {{ ?dataset dct:modified ?modified . }}
    FILTER(
        (BOUND(?created) && YEAR(?created) >= {year}) ||
        (BOUND(?modified) && YEAR(?modified) >= {year})
    )
}}
ORDER BY DESC(?modified) DESC(?created)
LIMIT {limit}"""

    def _get_statistics(self, params: Dict[str, Any]) -> str:
        """Generate query to get statistics."""
        return """SELECT 
    (COUNT(DISTINCT ?dataset) AS ?numDatasets)
    (COUNT(DISTINCT ?distribution) AS ?numDistributions)
    (COUNT(DISTINCT ?publisher) AS ?numPublishers)
WHERE {
    ?dataset a dcat:Dataset .
    OPTIONAL { ?dataset dcat:distribution ?distribution . }
    OPTIONAL { ?dataset dct:publisher ?publisher . }
}"""

    def _list_graphs(self, params: Dict[str, Any]) -> str:
        """Generate query to list all named graphs."""
        return """SELECT DISTINCT ?graph
WHERE {
    GRAPH ?graph {
        ?s ?p ?o .
    }
}"""

    def _list_services(self, params: Dict[str, Any]) -> str:
        """Generate query to list data services."""
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?service ?title ?endpointURL
WHERE {{
    ?service a dcat:DataService .
    OPTIONAL {{ ?service dct:title ?title . }}
    OPTIONAL {{ ?service dcat:endpointURL ?endpointURL . }}
}}
LIMIT {limit}"""

    def _list_publishers(self, params: Dict[str, Any]) -> str:
        """Generate query to list publishers."""
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?publisher ?name
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:publisher ?publisher .
    ?publisher foaf:name ?name .
}}
LIMIT {limit}"""

    def _list_properties(self, params: Dict[str, Any]) -> str:
        """Generate query to list dataset properties."""
        return """SELECT DISTINCT ?property
WHERE {
    ?dataset a dcat:Dataset .
    ?dataset ?property ?value .
}"""

    def _list_classes(self, params: Dict[str, Any]) -> str:
        """Generate query to list all classes."""
        return """SELECT DISTINCT ?class
WHERE {
    ?entity a ?class .
}
ORDER BY ?class"""

    def _list_publisher_properties(self, params: Dict[str, Any]) -> str:
        """Generate query to list publisher properties."""
        return """SELECT DISTINCT ?property
WHERE {
    ?dataset a dcat:Dataset ;
             dct:publisher ?publisher .
    ?publisher ?property ?value .
}"""

    def _list_service_properties(self, params: Dict[str, Any]) -> str:
        """Generate query to list data service properties."""
        return """SELECT DISTINCT ?property
WHERE {
    ?service a dcat:DataService .
    ?service ?property ?value .
}"""

    def _rank_publishers(self, params: Dict[str, Any]) -> str:
        """Generate query to rank publishers by dataset count."""
        limit = params.get("limit", 10)
        return f"""SELECT DISTINCT ?publisher ?name (COUNT(?dataset) AS ?numDatasets)
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:publisher ?publisher .
    ?publisher foaf:name ?name .
}}
GROUP BY ?publisher ?name
ORDER BY DESC(?numDatasets)
LIMIT {limit}"""

    def _rank_publishers_services(self, params: Dict[str, Any]) -> str:
        """Generate query to rank publishers by service count."""
        limit = params.get("limit", 10)
        return f"""SELECT DISTINCT ?publisher ?name (COUNT(?service) AS ?numServices)
WHERE {{
    ?service a dcat:DataService ;
             dct:publisher ?publisher .
    ?publisher foaf:name ?name .
}}
GROUP BY ?publisher ?name
ORDER BY DESC(?numServices)
LIMIT {limit}"""

    def _rank_publishers_combined(self, params: Dict[str, Any]) -> str:
        """Generate query to rank publishers by total resources."""
        limit = params.get("limit", 10)
        return f"""SELECT ?name 
       (COUNT(DISTINCT ?dataset) AS ?numDatasets)
       (COUNT(DISTINCT ?service) AS ?numDataServices)
       ((COUNT(DISTINCT ?dataset) + COUNT(DISTINCT ?service)) AS ?numTotal)
WHERE {{
    ?publisher foaf:name ?name .
    OPTIONAL {{ ?dataset a dcat:Dataset ; dct:publisher ?publisher . }}
    OPTIONAL {{ ?service a dcat:DataService ; dct:publisher ?publisher . }}
    FILTER(BOUND(?dataset) || BOUND(?service))
}}
GROUP BY ?name
ORDER BY DESC(?numTotal)
LIMIT {limit}"""

    def _count_by_graph(self, params: Dict[str, Any]) -> str:
        """Generate query to count resources by graph."""
        return """SELECT ?graph
       (COUNT(DISTINCT ?dataset) AS ?numDatasets)
       (COUNT(DISTINCT ?service) AS ?numDataServices)
WHERE {
    GRAPH ?graph {
        OPTIONAL { ?dataset a dcat:Dataset . }
        OPTIONAL { ?service a dcat:DataService . }
    }
}
GROUP BY ?graph
ORDER BY DESC(?numDatasets)"""

    def _filter_by_graph(self, params: Dict[str, Any]) -> str:
        """Generate query to filter by specific graph."""
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?dataset ?title
WHERE {{
    GRAPH <http://datos.gob.es/catalogo> {{
        ?dataset a dcat:Dataset ;
                 dct:title ?title .
    }}
}}
LIMIT {limit}"""

    def _filter_publishers_language(self, params: Dict[str, Any]) -> str:
        """Generate query to filter publishers by language."""
        language = params.get("language", "es")
        limit = params.get("limit", 100)
        return f"""SELECT DISTINCT ?publisher ?name
WHERE {{
    ?dataset a dcat:Dataset ;
             dct:publisher ?publisher .
    ?publisher foaf:name ?name .
    FILTER(LANGMATCHES(LANG(?name), "{language}"))
}}
LIMIT {limit}"""

    def _default_query(self) -> str:
        """Default fallback query."""
        return """SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o .
}
LIMIT 10"""
