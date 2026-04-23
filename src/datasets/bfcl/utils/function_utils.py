import re
import json


def sanitize_function_names(functions):
    """Replace dots with underscores in function names to comply with API requirements.
    
    This follows the same logic as the utils.convert_to_tool function from the original
    berkeley-function-call-leaderboard codebase.
    
    Args:
        functions: List of function dictionaries, each with a "name" field
        
    Returns:
        List of functions with sanitized names (dots replaced with underscores)
    """
    for function in functions:
        if "." in function["name"]:
            # OAI does not support "." in the function name so we replace it with "_". 
            # ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
            function["name"] = re.sub(r"\.", "_", function["name"])
    return functions 


def format_function_call(item):
    """Convert dict format to readable function call format.
    
    Converts representations like {'func_name': '{"n": 20}'} to func_name(n=20).
    
    Args:
        item: Either a dict with function names as keys and params as values,
              or a string representation of a function call
              
    Returns:
        String representation of the function call with keyword arguments
    """
    if isinstance(item, dict):
        # Convert {'func_name': '{params}'} to func_name(params)
        parts = []
        for func_name, params in item.items():
            if params == '{}' or params == '':
                parts.append(f"{func_name}()")
            else:
                # Try to parse params as JSON and format as keyword args
                try:
                    if isinstance(params, str):
                        param_dict = json.loads(params)
                        if isinstance(param_dict, dict):
                            # Format as keyword arguments
                            kwargs = ", ".join(f"{k}={v}" for k, v in param_dict.items())
                            parts.append(f"{func_name}({kwargs})")
                        else:
                            parts.append(f"{func_name}({params})")
                    else:
                        parts.append(f"{func_name}({params})")
                except json.JSONDecodeError:
                    # If it's not valid JSON, just use as is
                    parts.append(f"{func_name}({params})")
        return ", ".join(parts)
    else:
        return str(item)


def format_function_definition(func_spec):
    """Format a function specification into readable documentation.
    
    Creates a multi-line string showing the function name, description, and
    detailed parameter information including types and requirements.
    
    Args:
        func_spec: Dictionary containing function specification with keys:
                  - name: function name
                  - description: function description
                  - parameters: parameter schema dict
                  
    Returns:
        String with formatted function documentation
    """
    name = func_spec.get("name", "")
    description = func_spec.get("description", "")
    parameters = func_spec.get("parameters", {})
    
    # Format parameters nicely
    if isinstance(parameters, dict):
        props = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        param_lines = []
        for param_name, param_info in props.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            required_marker = " (required)" if param_name in required else " (optional)"
            param_lines.append(f"  - {param_name}: {param_type}{required_marker} - {param_desc}")
        
        if param_lines:
            param_str = f"Parameters:\n" + "\n".join(param_lines)
        else:
            param_str = "Parameters: No parameter information"
    else:
        param_str = "Parameters: No parameter information"
    
    return f"""Function: {name}
Description: {description or 'No description provided'}
{param_str}"""


def functions_are_identical(tools1, tools2):
    """Compare if two sets of function tools are identical.

    Returns True if all function names and descriptions match exactly.
    Parameters and other fields are not compared since they shouldn't change.

    Args:
        tools1: List of function tool dictionaries from first config
        tools2: List of function tool dictionaries from second config

    Returns:
        bool: True if functions are identical, False otherwise
    """
    if not tools1 and not tools2:
        return True
    if not tools1 or not tools2:
        return False
    if len(tools1) != len(tools2):
        return False

    # Sort by name for consistent comparison
    sorted1 = sorted(tools1, key=lambda x: x.get('name', ''))
    sorted2 = sorted(tools2, key=lambda x: x.get('name', ''))

    for t1, t2 in zip(sorted1, sorted2):
        # Compare function names
        if t1.get('name') != t2.get('name'):
            return False
        # Compare descriptions (the main thing that changes in iterations)
        if t1.get('description') != t2.get('description'):
            return False

    return True