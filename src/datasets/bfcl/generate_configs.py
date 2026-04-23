"""
Generate tool configuration files for BFCL tests.

Since each BFCL test has its own set of functions, we generate configs dynamically
from the test data.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import random
import string
import sys

# Add vendor directory to path
sys.path.insert(0, 'src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard')

from src.datasets.bfcl.utils.name_utils import greek_lowercase, animals, common_nouns_500
from openfunctions_evaluation import (
    parse_test_category_argument,
    TEST_FILE_MAPPING
)

class ModificationConfig:
    """Configuration for function modifications."""
    
    def __init__(self,
                 name_modifications: Optional[str] = None,
                 description_modifications: Optional[str] = None,
                 parameter_modifications: Optional[str] = None,
                 custom_descriptions: Optional[Dict[str, str]] = None,
                 seed: int = 42):
        self.name_mods = self._parse_modification_spec(name_modifications) if name_modifications else {}
        self.desc_mods = self._parse_modification_spec(description_modifications) if description_modifications else {}
        self.param_mods = self._parse_modification_spec(parameter_modifications) if parameter_modifications else {}
        self.custom_descriptions = custom_descriptions or {}
        self.seed = seed
        random.seed(seed)
        
        # Track name mappings for consistency
        self.name_mappings = {}
        self.used_names = set()
        # Track counters per test_id for increasing_number modification
        self.test_counters = {}
    
    def _parse_modification_spec(self, spec: str) -> Dict[str, str]:
        """Parse modification specification string.
        
        Format: "all:upper" or "0:lower,1:prefix:test_" or "random_3:upper"
        """
        if not spec:
            return {}
        
        result = {}
        parts = spec.split(',')
        
        for part in parts:
            if ':' not in part:
                continue
                
            components = part.split(':', 1)  # Split only on first colon
            if len(components) >= 2:
                indices = components[0]
                mod_spec = components[1]  # This includes everything after first colon
                
                if indices == 'all':
                    result['all'] = mod_spec
                elif indices.startswith('random_'):
                    # Handle random selection
                    count = int(indices.replace('random_', ''))
                    result['random'] = (count, mod_spec)  # Keep tuple for random
                else:
                    # Specific index
                    try:
                        idx = int(indices)
                        result[idx] = mod_spec
                    except ValueError:
                        pass
        
        return result
    
    def modify_name(self, name: str, index: int, test_id: str) -> str:
        """Apply name modification.
        
        Args:
            name: Original function name
            index: Index of function within the test (0, 1, 2, etc.)
            test_id: Unique identifier for the test
        """
        # Check if we already have a mapping for this name
        if name in self.name_mappings:
            return self.name_mappings[name]
        
        # Determine which modification to apply
        mod_spec = None
        if 'all' in self.name_mods:
            mod_spec = self.name_mods['all']
        elif index in self.name_mods:
            mod_spec = self.name_mods[index]
        elif 'random' in self.name_mods:
            count, mod_spec = self.name_mods['random']
            # Random selection logic would go here
            if random.random() > count / 100:  # If count is percentage
                return name  # Don't modify this one
        
        if not mod_spec:
            return name
        
        # Parse mod_spec if it contains a colon (e.g., "prefix:test_")
        if ':' in mod_spec:
            mod_type, mod_value = mod_spec.split(':', 1)
        else:
            mod_type = mod_spec
            mod_value = None
        
        # Apply modification
        new_name = name
        if mod_type == 'upper':
            new_name = name.upper()
        elif mod_type == 'lower':
            new_name = name.lower()
        elif mod_type == 'prefix' and mod_value:
            new_name = f"{mod_value}{name}"
        elif mod_type == 'suffix' and mod_value:
            new_name = f"{name}{mod_value}"
        elif mod_type == 'random':
            new_name = self._generate_random_name()
        elif mod_type == 'greek':
            new_name = self._sample_unique(greek_lowercase)
        elif mod_type == 'animal':
            new_name = self._sample_unique(animals)
        elif mod_type == 'noun':
            new_name = self._sample_unique(common_nouns_500)
        elif mod_type == 'increasing_number':
            # For increasing_number, use the index within the current test
            # This ensures each test has function_1, function_2, etc.
            new_name = f"function_{index + 1}"
        
        self.used_names.add(new_name)
        
        self.name_mappings[name] = new_name
        
        return new_name
    
    def _generate_random_name(self) -> str:
        """Generate a random function name."""
        length = random.randint(5, 15)
        return ''.join(random.choices(string.ascii_lowercase + '_', k=length))

    def _sample_unique(self, pool: list) -> str:
        """Randomly sample a name from pool that hasn't been used yet."""
        available = [name for name in pool if name not in self.used_names]
        if not available:
            available = pool
        return random.choice(available)
    
    def get_readable_config(self) -> str:
        """Generate a human-readable configuration string."""
        parts = []
        
        # Name modifications
        if self.name_mods:
            name_parts = []
            for key, value in self.name_mods.items():
                if key == 'all':
                    name_parts.append(f"all:{value}")
                elif key == 'random':
                    count, mod_spec = value
                    name_parts.append(f"random_{count}:{mod_spec}")
                elif isinstance(key, int):
                    name_parts.append(f"{key}:{value}")
            if name_parts:
                parts.append(f"name[{','.join(name_parts)}]")
        
        # Description modifications
        if self.desc_mods:
            desc_parts = []
            for key, value in self.desc_mods.items():
                if key == 'all':
                    desc_parts.append(f"all:{value}")
                elif key == 'random':
                    count, mod_spec = value
                    desc_parts.append(f"random_{count}:{mod_spec}")
                elif isinstance(key, int):
                    desc_parts.append(f"{key}:{value}")
            if desc_parts:
                parts.append(f"desc[{','.join(desc_parts)}]")
        
        # Parameter modifications
        if self.param_mods:
            param_parts = []
            for key, value in self.param_mods.items():
                if key == 'all':
                    param_parts.append(f"all:{value}")
                elif key == 'random':
                    count, mod_spec = value
                    param_parts.append(f"random_{count}:{mod_spec}")
                elif isinstance(key, int):
                    param_parts.append(f"{key}:{value}")
            if param_parts:
                parts.append(f"param[{','.join(param_parts)}]")
        
        if self.seed != 42:  # Only include seed if not default
            parts.append(f"seed{self.seed}")
        
        return "_".join(parts) if parts else "nomod"
    
    def modify_description(self, desc: str, name: str, index: int) -> str:
        """Apply description modification."""
        # Check for custom description first
        if name in self.custom_descriptions:
            return self.custom_descriptions[name]
        
        # Determine which modification to apply
        mod_spec = None
        if 'all' in self.desc_mods:
            mod_spec = self.desc_mods['all']
        elif index in self.desc_mods:
            mod_spec = self.desc_mods[index]
        
        if not mod_spec:
            return desc
        
        mod_type = mod_spec.split(':')[0] if ':' in mod_spec else mod_spec
        
        if mod_type == 'blank':
            return ""
        elif mod_type == 'minimal':
            return "An unknown function"
        elif mod_type == 'truncate':
            return desc[:20] + "..." if len(desc) > 20 else desc
        elif mod_type == 'upper':
            return desc.upper()
        elif mod_type == 'lower':
            return desc.lower()
        elif mod_type == 'reverse':
            return desc[::-1]
        elif mod_type == 'duplicate':
            return f"{desc} {desc}"
        elif mod_type == 'noise':
            # Add random characters
            noise = ''.join(random.choices(string.ascii_letters, k=5))
            return f"{desc} {noise}"
        
        return desc
    
    def modify_parameters(self, params: Dict, index: int) -> Dict:
        """Apply parameter modification."""
        # Determine which modification to apply
        mod_spec = None
        if 'all' in self.param_mods:
            mod_spec = self.param_mods['all']
        elif index in self.param_mods:
            mod_spec = self.param_mods[index]
        
        if not mod_spec:
            return params
        
        mod_type = mod_spec.split(':')[0] if ':' in mod_spec else mod_spec
        
        # Apply modification
        if mod_type == 'remove_all' or mod_type == 'none':
            # Remove all parameters - return None to signal removal
            return None
        elif mod_type == 'blank_descriptions':
            # Remove descriptions from parameters
            if params and 'properties' in params:
                for prop in params['properties'].values():
                    if 'description' in prop:
                        prop['description'] = ""
        elif mod_type == 'type_only':
            # Keep only type information
            if params and 'properties' in params:
                for prop in params['properties'].values():
                    keys_to_keep = ['type']
                    for key in list(prop.keys()):
                        if key not in keys_to_keep:
                            del prop[key]
        elif mod_type == 'generic_descriptions':
            # Replace with generic descriptions
            if params and 'properties' in params:
                for param_name, prop in params['properties'].items():
                    prop['description'] = f"The {param_name} parameter"
        
        return params

def fix_json_schema_types(schema: Any) -> Any:
    """Convert Python type names to valid JSON Schema types.
    
    Recursively fixes all type fields in a schema to use JSON Schema compliant types.
    """
    TYPE_MAPPING = {
        'float': 'number',
        'dict': 'object',
        'list': 'array',
        'tuple': 'array',
        'int': 'integer',
        'str': 'string',
        'bool': 'boolean',
        'None': 'null',
        'NoneType': 'null'
    }
    
    if isinstance(schema, dict):
        # Fix the type field if present
        if "type" in schema:
            if isinstance(schema["type"], str):
                schema["type"] = TYPE_MAPPING.get(schema["type"], schema["type"])
            elif isinstance(schema["type"], list):
                # Handle union types
                schema["type"] = [TYPE_MAPPING.get(t, t) if isinstance(t, str) else t for t in schema["type"]]
        
        # Recursively fix nested schemas
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema["properties"][prop_name] = fix_json_schema_types(prop_schema)
        
        if "items" in schema:
            schema["items"] = fix_json_schema_types(schema["items"])
            
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            schema["additionalProperties"] = fix_json_schema_types(schema["additionalProperties"])
    
    return schema


def generate_test_config(test_item: Dict, test_id: int, modification_config: Optional[ModificationConfig] = None) -> Dict:
    """Generate a configuration for a single test."""
    # Clear name mappings for each test to ensure clean function numbering
    if modification_config:
        modification_config.name_mappings.clear()
    
    functions = test_item.get("function", [])
    
    # Ensure functions is a list
    if isinstance(functions, dict):
        functions = [functions]
    elif isinstance(functions, str):
        # If it's a string, parse it as JSON
        functions = [json.loads(functions)]
    
    # Apply modifications if specified
    if modification_config:
        modified_functions = []
        name_mapping = {}
        
        for i, func in enumerate(functions):
            # Make a copy of the function
            if isinstance(func, dict):
                modified_func = func.copy()
            else:
                # If it's a string, parse it first
                modified_func = json.loads(func) if isinstance(func, str) else func
            original_name = modified_func.get("name", "")
            
            # Modify name
            new_name = modification_config.modify_name(original_name, i, str(test_id))
            modified_func["name"] = new_name
            name_mapping[original_name] = new_name
            
            # Modify description
            if "description" in modified_func:
                modified_func["description"] = modification_config.modify_description(
                    modified_func["description"], new_name, i
                )
            
            # Modify parameters
            if "parameters" in modified_func:
                modified_params = modification_config.modify_parameters(
                    modified_func["parameters"], i
                )
                if modified_params is None:
                    # Remove parameters entirely
                    del modified_func["parameters"]
                else:
                    # Fix JSON Schema types after modification
                    modified_func["parameters"] = fix_json_schema_types(modified_params)
            
            modified_functions.append(modified_func)
        
        functions = modified_functions
    else:
        # Even without modifications, fix JSON Schema types
        for func in functions:
            if "parameters" in func:
                func["parameters"] = fix_json_schema_types(func["parameters"])
        name_mapping = {f["name"]: f["name"] for f in functions}
    
    config = {
        "test_id": test_id,
        "question": test_item.get("question", ""),
        "tools": functions,
        "ground_truth": test_item.get("ground_truth", []),
        "name_mapping": name_mapping,  # Original -> Modified name mapping
        "metadata": {
            "category": test_item.get("category", ""),
            "difficulty": test_item.get("difficulty", ""),
        }
    }
    
    # Add execution results if available
    if "execution_result" in test_item:
        config["execution_result"] = test_item["execution_result"]
    if "execution_result_type" in test_item:
        config["execution_result_type"] = test_item["execution_result_type"]
    
    return config


def generate_category_configs(
    test_category: str,
    output_dir: Path,
    modification_config: Optional[ModificationConfig] = None,
    config_name: Optional[str] = None
) -> Dict:
    """Generate configs for all tests in a category."""
    
    # Parse test category to get the actual file name
    test_name_list, test_filename_list = parse_test_category_argument([test_category])
    
    if not test_filename_list:
        raise ValueError(f"No file mapping for category: {test_category}")
    
    # Use the first file from the list (should only be one for a single category)
    file_to_open = test_filename_list[0]
    
    # Load test data
    full_path = Path("src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard/data") / file_to_open
    test_data = []
    with open(full_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    
    # Generate config for each test
    configs = []
    for i, test_item in enumerate(test_data):
        config = generate_test_config(test_item, i, modification_config)
        configs.append(config)
    
    # Generate config name with modification details
    if not config_name:
        if modification_config and (modification_config.name_mods or modification_config.desc_mods or modification_config.param_mods):
            readable_config = modification_config.get_readable_config()
            generated_config_name = f"{test_category}_{readable_config}"
        else:
            generated_config_name = f"{test_category}_base"
    else:
        generated_config_name = config_name
    
    summary = {
        "config_name": generated_config_name,
        "config_description": f"Tool configurations for {test_category} tests",
        "test_category": test_category,
        "num_tests": len(configs),
        "modifications": {
            "names": modification_config.name_mods if modification_config else {},
            "descriptions": modification_config.desc_mods if modification_config else {},
            "parameters": modification_config.param_mods if modification_config else {}
        } if modification_config else None,
        "tests": configs
    }
    
    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use the test name (without .json extension) for the config file
    test_name = test_name_list[0] if test_name_list else test_category
    
    # Build filename with modification details
    if modification_config and (modification_config.name_mods or modification_config.desc_mods or modification_config.param_mods):
        readable_config = modification_config.get_readable_config()
        config_filename = f"{test_name}_{readable_config}_config.json"
    else:
        config_filename = f"{test_name}_base_config.json"
    
    config_file = output_dir / config_filename
    
    with open(config_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated config for {test_name}: {config_file}")
    print(f"  - {len(configs)} test configurations")
    if modification_config and modification_config.name_mappings:
        print(f"  - {len(modification_config.name_mappings)} name modifications")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate BFCL tool configurations")
    parser.add_argument("--test-category", type=str, nargs="+", default=["simple"], help="Test categories to generate configs for")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for configs")
    parser.add_argument("--config-name", type=str, default=None, help="Name for the configuration set")
    parser.add_argument("--name-modifications", type=str, default=None, help="Name modifications (e.g., 'all:upper' or '0:lower,1:prefix:test_')")
    parser.add_argument("--description-modifications", type=str, default=None, help="Description modifications (e.g., 'all:minimal')")
    parser.add_argument("--parameter-modifications", type=str, default=None, help="Parameter modifications (e.g., 'all:blank_descriptions')")
    parser.add_argument("--custom-descriptions", type=str, default=None, help="JSON file with custom descriptions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for modifications")
    args = parser.parse_args()
    
    # Load custom descriptions if provided
    custom_descriptions = None
    if args.custom_descriptions:
        with open(args.custom_descriptions, 'r') as f:
            custom_descriptions = json.load(f)
    
    # Create modification config if any modifications specified
    modification_config = None
    if any([args.name_modifications, args.description_modifications, 
            args.parameter_modifications, custom_descriptions]):
        modification_config = ModificationConfig(
            name_modifications=args.name_modifications,
            description_modifications=args.description_modifications,
            parameter_modifications=args.parameter_modifications,
            custom_descriptions=custom_descriptions,
            seed=args.seed
        )
    
    output_dir = Path(args.output_dir)
    
    # Generate configs for each category
    for category in args.test_category:
        generate_category_configs(category, output_dir, modification_config, args.config_name)

if __name__ == "__main__":
    main()