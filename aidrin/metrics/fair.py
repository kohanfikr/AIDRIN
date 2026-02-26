from typing import Dict, Any, List

class FAIRCompliance:
    """Evaluates metadata compliance against FAIR (Findable, Accessible, Interoperable, Reusable) principles."""
    
    # Based on DCAT/DataCite structure mentioned in the paper
    FAIR_KEYS = {
        "findable": ["identifier", "title", "description", "keyword", "theme", "landingPage"],
        "accessible": ["distribution", "downloadURL", "format", "accessLevel", "publisher"],
        "interoperable": ["format", "conformsTo", "references"],
        "reusable": ["license", "programCode", "bureauCode", "conformsTo", "description", "format"]
    }
    
    @staticmethod
    def evaluate(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates FAIR compliance score from a metadata dictionary."""
        if not metadata:
            return {"error": "No metadata provided."}
            
        results = {
            "findable": {"score": 0, "total": len(FAIRCompliance.FAIR_KEYS["findable"]), "found": []},
            "accessible": {"score": 0, "total": len(FAIRCompliance.FAIR_KEYS["accessible"]), "found": []},
            "interoperable": {"score": 0, "total": len(FAIRCompliance.FAIR_KEYS["interoperable"]), "found": []},
            "reusable": {"score": 0, "total": len(FAIRCompliance.FAIR_KEYS["reusable"]), "found": []}
        }
        
        # Flatten metadata loosely for key checking
        def get_all_keys(d, keys=set()):
            for k, v in d.items():
                keys.add(k)
                if isinstance(v, dict):
                    get_all_keys(v, keys)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                             get_all_keys(item, keys)
            return keys
            
        provided_keys = get_all_keys(metadata)
        
        total_possible = 0
        total_found = 0
        
        for category, target_keys in FAIRCompliance.FAIR_KEYS.items():
             for key in target_keys:
                 total_possible += 1
                 # Using lower for case-insensitive matching
                 if any(key.lower() == pk.lower() for pk in provided_keys):
                     results[category]["score"] += 1
                     results[category]["found"].append(key)
                     total_found += 1
                     
        overall_score = float(total_found) / total_possible if total_possible > 0 else 0.0
        
        return {
            "overall_compliance_score": overall_score,
            "category_breakdown": results
        }
