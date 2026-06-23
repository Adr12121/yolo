import os
import json
from typing import List, Dict, Set

class FiliationEngine:
    """
    Moteur de recherche généalogique des parcelles cadastrales (Filiation).
    Utilise le graphe pré-calculé (JSON) à partir des données DFI de la DGFiP.
    
    L'algorithme parcourt le graphe de manière récursive pour trouver les parcelles "feuilles"
    (celles qui n'ont pas été divisées à leur tour).
    """

    def __init__(self, dfi_json_path: str = None):
        self.dfi_json_path = dfi_json_path
        self._graph = None
        
        # Si un fichier est fourni à l'initialisation, on le charge en mémoire.
        if dfi_json_path and os.path.exists(dfi_json_path):
            self.load_data(dfi_json_path)

    def load_data(self, path: str):
        """Charge le graphe JSON en mémoire."""
        try:
            print(f"Chargement du graphe de filiation DFI: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                self._graph = json.load(f)
            print(f"Graphe chargé avec succès ({len(self._graph)} noeuds).")
        except Exception as e:
            print(f"Erreur lors du chargement du graphe DFI: {e}")
            self._graph = {}

    def _normalize_numero(self, numero: str) -> str:
        """Normalise un numéro de parcelle (ex: '14' -> '0014' ou juste '14' selon la base)."""
        if not numero: return ""
        return str(numero).zfill(4)

    def trouver_parcelles_filles_directes(self, code_commune: str, section: str, numero: str) -> List[Dict]:
        """Retourne les parcelles issues directement d'une division/fusion (Niveau 1)."""
        if not self._graph:
            return []
            
        numero_norm = self._normalize_numero(numero)
        section_norm = str(section).strip().upper().lstrip('0')
        if not section_norm: # Fallback si jamais c'était '00'
            section_norm = str(section).strip().upper()
        
        noeud_id = f"{code_commune}_{section_norm}_{numero_norm}"
        return self._graph.get(noeud_id, [])

    def trouver_parcelles_actuelles(self, code_commune: str, section: str, numero: str, max_depth: int = 10) -> List[Dict]:
        """
        Algorithme récursif (Parcours en profondeur) pour trouver les parcelles "feuilles".
        Retourne la liste des parcelles finales actuellement en vigueur.
        """
        if not self._graph:
            print("⚠️ Aucun graphe DFI chargé. Impossible de tracer la filiation.")
            return []

        feuilles = []
        visited = set()
        
        def dfs(c_com, c_sec, c_num, depth):
            if depth > max_depth:
                return # Sécurité anti-boucle infinie
                
            c_num_norm = self._normalize_numero(c_num)
            c_sec_norm = str(c_sec).strip().upper().lstrip('0')
            if not c_sec_norm:
                c_sec_norm = str(c_sec).strip().upper()
            noeud_id = f"{c_com}_{c_sec_norm}_{c_num_norm}"
            
            if noeud_id in visited:
                return
            visited.add(noeud_id)
            
            filles = self._graph.get(noeud_id, [])
            
            if not filles:
                # C'est une parcelle feuille (pas de division ultérieure dans la base)
                feuilles.append({
                    "code_commune": c_com,
                    "section": c_sec_norm,
                    "numero": c_num_norm
                })
            else:
                for fille in filles:
                    dfs(fille["code_commune"], fille["section"], fille["numero"], depth + 1)

        dfs(str(code_commune), str(section), str(numero), 0)
        
        # Dédoublonnage
        resultats_uniques = [dict(t) for t in {tuple(d.items()) for d in feuilles}]
        return resultats_uniques

# Instance globale (Singleton) pour une utilisation simple
_engine_instance = None

def get_filiation_engine(dfi_path: str = None) -> FiliationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FiliationEngine(dfi_path)
    return _engine_instance
