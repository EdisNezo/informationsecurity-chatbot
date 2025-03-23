import re
from typing import Dict, Tuple, List, Any, Optional
from langchain.prompts import PromptTemplate

class HallucinationManager:
    """
    Klasse zur Erkennung und Behandlung von Halluzinationen in generiertem Content.
    Kombiniert regelbasierte und LLM-basierte Ansätze zur Qualitätssicherung.
    """
    
    def __init__(self, llm_chain_provider):
        """
        Initialisiert den HallucinationManager.
        
        Args:
            llm_chain_provider: Ein Objekt, das Zugriff auf die benötigten LLM-Chains bietet
        """
        self.chains = {
            "hallucination_check": llm_chain_provider.create_chain(
                self._create_hallucination_check_prompt()
            ),
            "content_correction": llm_chain_provider.create_chain(
                self._create_content_correction_prompt()
            )
        }
    
    def comprehensive_hallucination_detection(self, content: str, user_input: str, context_text: str) -> Dict[str, Any]:
        """
        Führt eine umfassende Halluzinationserkennung durch, die Musterprüfung und 
        kontextbasierte Analyse kombiniert.
        
        Args:
            content: Zu prüfender Inhalt
            user_input: Ursprüngliche Eingabe des Nutzers
            context_text: Kontextinformationen aus dem Retrieval
            
        Returns:
            Dictionary mit detaillierten Analyseergebnissen und korrigiertem Inhalt
        """
        results = {
            "has_issues": False,
            "corrected_content": content,
            "detected_patterns": {},
            "confidence_score": 1.0,
            "suspicious_sections": [],
            "llm_analysis": {}
        }
        
        # 1. Musterbasierte Analyse
        hallucination_patterns = {
            "Unsicherheit": [
                r"könnte sein", r"möglicherweise", r"eventuell", r"vielleicht",
                r"unter umständen", r"es ist denkbar", r"in der regel"
            ],
            "Widersprüche": [
                r"einerseits.*andererseits", r"jedoch", r"allerdings",
                r"im gegensatz dazu", r"wiederum"
            ],
            "Vage Aussagen": [
                r"irgendwie", r"gewissermaßen", r"im großen und ganzen",
                r"im allgemeinen", r"mehr oder weniger"
            ]
        }
        
        # Überprüfe den Text auf Muster
        content_lower = content.lower()
        
        for category, patterns in hallucination_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    start_pos = max(0, match.start() - 40)
                    end_pos = min(len(content_lower), match.end() + 40)
                    context = content_lower[start_pos:end_pos]
                    category_matches.append({
                        "pattern": pattern,
                        "context": context
                    })
                    
                    # Reduziere den Confidence-Score für jeden Fund
                    results["confidence_score"] = max(0.1, results["confidence_score"] - 0.05)
                    
                    # Speichere den Abschnitt als verdächtig
                    results["suspicious_sections"].append(context)
            
            if category_matches:
                results["detected_patterns"][category] = category_matches
        
        # 2. LLM-basierte Analyse mithilfe eines spezialisierten Prompts
        llm_response = self.chains["hallucination_check"].run({
            "content": content,
            "user_input": user_input,
            "context_text": context_text
        })
        
        # Speichere die LLM-Analyse im Ergebnis
        results["llm_analysis"]["raw_response"] = llm_response
        results["has_issues"] = "KEINE_PROBLEME" not in llm_response
        
        # 3. Kombinierte Ergebnisauswertung und Korrektur
        if results["has_issues"] or results["confidence_score"] < 0.7:
            # Setze das has_issues Flag, wenn entweder LLM oder Musteranalyse Probleme fand
            results["has_issues"] = True
            
            # Korrigiere den Inhalt basierend auf der LLM-Analyse
            if "KEINE_PROBLEME" not in llm_response:
                results["corrected_content"] = self.generate_content_with_corrections(content, llm_response)
            
            # Füge Informationen aus der Musteranalyse hinzu, wenn die LLM-Analyse keine Probleme fand
            elif results["confidence_score"] < 0.7:
                # Hier könnten zusätzliche Korrekturen basierend auf der Musteranalyse vorgenommen werden
                results["llm_analysis"]["warning"] = "Musterbasierte Analyse hat potenzielle Probleme gefunden, die von der LLM-Analyse nicht erkannt wurden."
        
        return results
    
    def check_hallucinations(self, content: str, user_input: str, context_text: str) -> Tuple[bool, str]:
        """
        Überprüft den generierten Inhalt auf Halluzinationen.
        
        Args:
            content: Generierter Inhalt
            user_input: Ursprüngliche Eingabe des Nutzers
            context_text: Kontextinformationen aus dem Retrieval
            
        Returns:
            Tuple aus (hat_probleme, korrigierter_inhalt)
        """
        response = self.chains["hallucination_check"].run({
            "content": content,
            "user_input": user_input,
            "context_text": context_text
        })
        
        # Prüfe, ob Probleme gefunden wurden
        has_issues = "KEINE_PROBLEME" not in response
        
        # Korrigiere den Inhalt basierend auf dem Check
        if has_issues:
            corrected_content = self.generate_content_with_corrections(content, response)
        else:
            corrected_content = content
        
        return has_issues, corrected_content
    
    def advanced_hallucination_detection(self, content: str) -> Dict[str, Any]:
        """
        Führt eine erweiterte Halluzinationserkennung durch.
        
        Args:
            content: Zu prüfender Inhalt
            
        Returns:
            Dictionary mit Analyseergebnissen
        """
        # Muster für typische Halluzinationsindikatoren
        hallucination_patterns = {
            "Unsicherheit": [
                r"könnte sein", r"möglicherweise", r"eventuell", r"vielleicht",
                r"unter umständen", r"es ist denkbar", r"in der regel"
            ],
            "Widersprüche": [
                r"einerseits.*andererseits", r"jedoch", r"allerdings",
                r"im gegensatz dazu", r"wiederum"
            ],
            "Vage Aussagen": [
                r"irgendwie", r"gewissermaßen", r"im großen und ganzen",
                r"im allgemeinen", r"mehr oder weniger"
            ]
        }
        
        results = {
            "detected_patterns": {},
            "confidence_score": 1.0,  # Anfangswert, wird für jedes gefundene Muster reduziert
            "suspicious_sections": []
        }
        
        # Überprüfe den Text auf Muster
        content_lower = content.lower()
        
        for category, patterns in hallucination_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    start_pos = max(0, match.start() - 40)
                    end_pos = min(len(content_lower), match.end() + 40)
                    context = content_lower[start_pos:end_pos]
                    category_matches.append({
                        "pattern": pattern,
                        "context": context
                    })
                    
                    # Reduziere den Confidence-Score für jeden Fund
                    results["confidence_score"] = max(0.1, results["confidence_score"] - 0.05)
                    
                    # Speichere den Abschnitt als verdächtig
                    results["suspicious_sections"].append(context)
            
            if category_matches:
                results["detected_patterns"][category] = category_matches
        
        return results
    
    def generate_content_with_corrections(self, original_content: str, hallucination_analysis: str) -> str:
        """
        Generiert korrigierten Inhalt basierend auf der Halluzinationsanalyse.
        
        Args:
            original_content: Der ursprüngliche Inhalt
            hallucination_analysis: Die Ergebnisse der Halluzinationsanalyse
            
        Returns:
            Korrigierter Inhalt
        """
        corrected_content = self.chains["content_correction"].run({
            "original_content": original_content,
            "hallucination_analysis": hallucination_analysis
        })
        
        return corrected_content
    
    def _create_hallucination_check_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Halluzinationsprüfung.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Überprüfe den folgenden Inhalt für einen E-Learning-Kurs zur Informationssicherheit auf mögliche Ungenauigkeiten oder Halluzinationen.
        
        Zu prüfender Text: 
        {content}
        
        Kontext aus der Kundenantwort:
        {user_input}
        
        Verfügbare Fachinformationen:
        {context_text}
        
        Bitte identifiziere:
        1. Aussagen über Informationssicherheit, die nicht durch die verfügbaren Fachinformationen gestützt werden
        2. Empfehlungen oder Maßnahmen, die für den beschriebenen Unternehmenskontext ungeeignet sein könnten
        3. Technische Begriffe oder Konzepte, die falsch verwendet wurden
        4. Widersprüche zu bewährten Sicherheitspraktiken
        5. Unzutreffende Behauptungen über Bedrohungen oder deren Auswirkungen
        
        Für jede identifizierte Problemstelle:
        - Zitiere die betreffende Textpassage
        - Erkläre, warum dies problematisch ist
        - Schlage eine fachlich korrekte Alternative vor
        
        Falls keine Probleme gefunden wurden, antworte mit "KEINE_PROBLEME".
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["content", "user_input", "context_text"]
        )
    
    def _create_content_correction_prompt(self) -> PromptTemplate:
        """
        Erstellt ein Prompt-Template für die Korrektur von Inhalten.
        
        Returns:
            PromptTemplate-Objekt
        """
        template = """
        Bitte korrigiere den folgenden Inhalt basierend auf der Halluzinationsanalyse:
        
        Ursprünglicher Inhalt:
        {original_content}
        
        Analyse der Probleme:
        {hallucination_analysis}
        
        Erstelle eine korrigierte Version des Inhalts, die alle identifizierten Probleme behebt.
        Achte darauf, die Korrekturen nahtlos in den Originaltext einzufügen, um einen kohärenten Gesamttext zu erzeugen.
        Der korrigierte Text sollte stilistisch dem Original entsprechen und Fachinformationen korrekt darstellen.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["original_content", "hallucination_analysis"]
        )
    
    def add_custom_pattern(self, category: str, pattern: str) -> None:
        """
        Fügt ein benutzerdefiniertes Muster zur Halluzinationserkennung hinzu.
        
        Args:
            category: Kategorie des Musters
            pattern: Regulärer Ausdruck als Muster
        """
        # Diese Methode würde in einer vollständigen Implementierung 
        # die Muster dynamisch verwalten
        pass
    
    def generate_hallucination_report(self, detection_results: Dict[str, Any]) -> str:
        """
        Generiert einen lesbaren Bericht aus den Erkennungsergebnissen.
        
        Args:
            detection_results: Ergebnisse der Halluzinationserkennung
            
        Returns:
            Formatierter Bericht
        """
        report = "# Halluzinationsanalyse-Bericht\n\n"
        
        # Gesamtbewertung
        confidence = detection_results.get("confidence_score", 0) * 100
        report += f"## Gesamtbewertung\n"
        report += f"Vertrauenswert: {confidence:.1f}%\n"
        report += f"Probleme gefunden: {'Ja' if detection_results.get('has_issues', False) else 'Nein'}\n\n"
        
        # Gefundene Muster
        if "detected_patterns" in detection_results and detection_results["detected_patterns"]:
            report += "## Gefundene Muster\n\n"
            for category, matches in detection_results["detected_patterns"].items():
                report += f"### {category}\n"
                for i, match in enumerate(matches, 1):
                    report += f"{i}. **Muster**: `{match['pattern']}`\n"
                    report += f"   **Kontext**: \"...{match['context']}...\"\n\n"
        
        # LLM-Analyse
        if "llm_analysis" in detection_results and detection_results["llm_analysis"]:
            report += "## LLM-Analyse\n\n"
            
            if "warning" in detection_results["llm_analysis"]:
                report += f"**Warnung**: {detection_results['llm_analysis']['warning']}\n\n"
            
            if "raw_response" in detection_results["llm_analysis"]:
                # Nur anzeigen, wenn es kein "KEINE_PROBLEME" ist
                if "KEINE_PROBLEME" not in detection_results["llm_analysis"]["raw_response"]:
                    report += "### Detaillierte Analyse\n"
                    report += detection_results["llm_analysis"]["raw_response"] + "\n\n"
        
        return report