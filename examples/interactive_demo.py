#!/usr/bin/env python3
"""
Interactive SPARQL Humanize Demo

This script provides an interactive demonstration of how SPARQL Humanize
converts natural language questions into SPARQL queries.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparql_humanize.predict import predict  # noqa: E402


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("  🤖 SPARQL HUMANIZE - Conversor de Lenguaje Natural a SPARQL")
    print("=" * 80)
    print()
    print("  Este demo muestra cómo tus preguntas se clasifican en tipos de consulta.")
    print("  En una aplicación real, estos tipos generarían consultas SPARQL completas.")
    print()
    print("=" * 80)
    print()


def explain_query_type(query_type: str) -> tuple[str, str]:
    """
    Explain what each query type does.
    
    Returns:
        Tuple of (spanish_explanation, example_sparql_snippet)
    """
    explanations = {
        "list_datasets": (
            "📋 LISTAR DATASETS: Muestra todos los datasets disponibles",
            "SELECT ?dataset ?title WHERE { ?dataset a dcat:Dataset }"
        ),
        "search_datasets": (
            "🔍 BUSCAR POR TEMA: Busca datasets sobre un tema específico",
            "SELECT ?dataset WHERE { ?dataset dcat:theme 'salud' }"
        ),
        "get_distribution": (
            "📄 FILTRAR POR FORMATO: Obtiene datasets en un formato específico (CSV, JSON, etc.)",
            "SELECT ?dataset WHERE { ?dataset dcat:mediaType 'text/csv' }"
        ),
        "get_metadata": (
            "ℹ️  OBTENER METADATOS: Muestra información detallada de un dataset",
            "SELECT ?title ?publisher ?modified WHERE { ... }"
        ),
        "filter_datasets": (
            "📅 FILTRAR POR FECHA: Busca datasets creados antes/después de una fecha",
            "SELECT ?dataset WHERE { ?dataset dct:issued > '2020-01-01' }"
        ),
        "get_statistics": (
            "📊 ESTADÍSTICAS: Cuenta datasets, agrupa por tema, etc.",
            "SELECT (COUNT(?dataset) as ?count) WHERE { ... }"
        )
    }
    
    return explanations.get(query_type, (
        "❓ TIPO DESCONOCIDO",
        "No hay ejemplo disponible"
    ))


def demo_questions():
    """Run demo with predefined questions."""
    questions = [
        ("What datasets are available?", "Inglés"),
        ("Muéstrame datos sobre salud", "Español"),
        ("I need CSV files", "Inglés"),
        ("Datasets creados después de 2020", "Español"),
        ("How many datasets do we have?", "Inglés"),
        ("Información del dataset de educación", "Español"),
    ]
    
    print("📚 EJEMPLOS DE PREGUNTAS:")
    print("-" * 80)
    print()
    
    for i, (question, lang) in enumerate(questions, 1):
        print(f"{i}. [{lang}] \"{question}\"")
        print("   " + "─" * 76)
        
        try:
            result = predict(question)
            intent = result["predicted_query_type"]
            confidence = result["confidence"]
            
            explanation, sparql_example = explain_query_type(intent)
            
            # Format confidence as bar
            bar_length = int(confidence * 40)
            confidence_bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"   ✓ Tipo detectado: {intent}")
            print(f"   {explanation}")
            print(f"   Confianza: [{confidence_bar}] {confidence:.1%}")
            print(f"   SPARQL: {sparql_example}")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
        
        print()
    
    print("=" * 80)
    print()


def interactive_mode():
    """Run interactive question-answer mode."""
    print("💬 MODO INTERACTIVO")
    print("=" * 80)
    print()
    print("  Escribe tus preguntas y ve cómo se clasifican.")
    print("  Ejemplos:")
    print("    - 'datasets sobre medio ambiente'")
    print("    - 'show me JSON files'")
    print("    - 'cuántos datasets hay'")
    print()
    print("  Escribe 'salir' o 'exit' para terminar.")
    print()
    print("-" * 80)
    print()
    
    while True:
        try:
            question = input("Tu pregunta ❯ ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n👋 ¡Hasta luego!\n")
                break
            
            print()
            result = predict(question)
            intent = result["predicted_query_type"]
            confidence = result["confidence"]
            
            explanation, sparql_example = explain_query_type(intent)
            
            # Confidence visualization
            bar_length = int(confidence * 50)
            confidence_bar = "█" * bar_length + "░" * (50 - bar_length)
            
            # Color coding based on confidence
            if confidence > 0.7:
                status = "🟢 Alta confianza"
            elif confidence > 0.4:
                status = "🟡 Confianza media"
            else:
                status = "🔴 Baja confianza (necesita más entrenamiento)"
            
            print(f"  Resultado: {status}")
            print(f"  Tipo: {intent}")
            print(f"  {explanation}")
            print(f"  Confianza: [{confidence_bar}] {confidence:.1%}")
            print(f"  SPARQL: {sparql_example}")
            print()
            print("-" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!\n")
            break
        except EOFError:
            print("\n\n👋 ¡Hasta luego!\n")
            break


def main():
    """Main entry point."""
    print_banner()
    
    # Check if model exists
    model_path = Path(__file__).parent.parent / "models" / "sparql_classifier.pkl"
    if not model_path.exists():
        print("❌ ERROR: Modelo no encontrado.")
        print()
        print("Por favor, entrena el modelo primero:")
        print("  pdm run python -m sparql_humanize.cli train --dataset data/dataset_extended.csv")
        print()
        return
    
    # Show demo questions
    demo_questions()
    
    # Ask if user wants interactive mode
    print("¿Quieres probar con tus propias preguntas? (s/n): ", end="")
    try:
        response = input().strip().lower()
        if response in ['s', 'y', 'si', 'yes']:
            print()
            interactive_mode()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 ¡Hasta luego!\n")


if __name__ == "__main__":
    main()
