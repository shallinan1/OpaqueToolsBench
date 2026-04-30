"""
Configurable tool handler that creates multiple searchers and tools based on JSON config.
Allows experimenting with different opacity levels and tool combinations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _serialize_search_output(results: List[Dict[str, Any]]) -> str:
    """Serialize search results and make empty results explicit to the model."""
    if results:
        return json.dumps(results, indent=2)
    return json.dumps(
        {
            "status": "no_results",
            "num_results": 0,
            "results": [],
            "message": (
                "No results found for this tool call. "
                "Try rephrasing the query or using a different search tool."
            ),
        },
        indent=2,
    )


class SearchToolHandler:
    """Local replacement for vendor SearchToolHandler — just a data holder."""

    def __init__(self, searcher, snippet_max_tokens: int | None = None,
                 k: int = 5, include_get_document: bool = False):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        self._add_snippets(candidates)
        return _serialize_search_output([self._format_candidate(c) for c in candidates])

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)

    def _add_snippets(self, candidates):
        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.snippet_max_tokens,
                )
                if len(tokens) == self.snippet_max_tokens:
                    cand["snippet"] = self.tokenizer.decode(tokens, skip_special_tokens=True)
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

    @staticmethod
    def _format_candidate(cand):
        result = {"docid": cand["docid"], "snippet": cand["snippet"]}
        if cand.get("score") is not None:
            result["score"] = cand["score"]
        return result


class DynamicFilteringSearchToolHandler:
    """
    Wrapper around SearchToolHandler that supports dynamic filtering parameters.
    """
    
    def __init__(self, base_handler: SearchToolHandler):
        self.base_handler = base_handler
        self._batch_search_enabled = True
        self._batch_search_failures = 0
        
        # Replace the base handler's methods with our dynamic versions
        base_handler._search = self._dynamic_search
        base_handler._get_document = self._dynamic_get_document
        base_handler.execute_batch_tools = self._execute_batch_tools
    
    def _dynamic_search(self, query: str, **kwargs):
        """Enhanced search that passes filter parameters to searcher."""
        # Extract filter parameters
        filter_category = kwargs.pop('filter_category', None)
        filter_domain = kwargs.pop('filter_domain', None)
        
        # Call searcher with dynamic filter parameters
        candidates = self.base_handler.searcher.search(
            query, 
            self.base_handler.k, 
            filter_category=filter_category,
            filter_domain=filter_domain
        )
        
        # Apply snippet processing (copied from original implementation)
        if self.base_handler.snippet_max_tokens and self.base_handler.snippet_max_tokens > 0 and self.base_handler.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.base_handler.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.base_handler.snippet_max_tokens,
                )
                if len(tokens) < self.base_handler.snippet_max_tokens:
                    cand["snippet"] = text
                else:
                    cand["snippet"] = self.base_handler.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]
        
        # Format results (copied from original implementation)
        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({
                    "docid": cand["docid"],
                    "snippet": cand["snippet"]
                })
            else:
                results.append({
                    "docid": cand["docid"],
                    "score": cand["score"],
                    "snippet": cand["snippet"]
                })
        
        return _serialize_search_output(results)
    
    def _dynamic_get_document(self, docid: str, **kwargs):
        """Enhanced get_document that passes filter parameters to searcher."""
        # Extract filter parameters
        filter_category = kwargs.pop('filter_category', None)
        filter_domain = kwargs.pop('filter_domain', None)
        
        # Call searcher with dynamic filter parameters
        result = self.base_handler.searcher.get_document(
            docid,
            filter_category=filter_category,
            filter_domain=filter_domain
        )
        
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)
    
    def _execute_batch_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Execute multiple tool calls, using batch search when beneficial."""
        
        # Separate search calls from other tool types
        search_calls = []
        search_indices = []
        other_calls = []
        
        for i, call in enumerate(tool_calls):
            if call['tool_name'] == 'search':
                search_calls.append(call)
                search_indices.append(i)
            else:
                other_calls.append((i, call))
        
        results = [None] * len(tool_calls)
        
        # Automatic detection: use batch search if multiple searches AND batch_search available
        if (
            len(search_calls) > 1
            and self._batch_search_enabled
            and hasattr(self.base_handler.searcher, 'batch_search')
        ):
            logger.info(f"Using batch search for {len(search_calls)} search calls")
            
            # Extract queries and filters
            queries = [call['arguments']['query'] for call in search_calls]
            filter_categories = [call['arguments'].get('filter_category') for call in search_calls]
            filter_domains = [call['arguments'].get('filter_domain') for call in search_calls]
            
            try:
                # Execute batch search
                batch_results = self.base_handler.searcher.batch_search(
                    queries, self.base_handler.k, filter_categories, filter_domains
                )
                
                # Format each result set and place in correct position
                for idx, raw_results in zip(search_indices, batch_results):
                    formatted = self._format_search_results(raw_results)
                    results[idx] = _serialize_search_output(formatted)
                    
            except Exception as e:
                self._batch_search_failures += 1
                error_text = str(e)
                is_jvm_failure = (
                    "JVM exception occurred" in error_text
                    or "NullPointerException" in error_text
                )
                if is_jvm_failure or self._batch_search_failures >= 3:
                    self._batch_search_enabled = False
                    logger.warning(
                        "Batch search disabled after %d failure(s); using sequential search for remaining calls. Last error: %s",
                        self._batch_search_failures,
                        error_text,
                    )
                else:
                    logger.warning(
                        "Batch search failed (%d), falling back to sequential this round: %s",
                        self._batch_search_failures,
                        error_text,
                    )
                # Fallback to sequential processing
                for idx, call in zip(search_indices, search_calls):
                    results[idx] = self.base_handler.execute_tool(call['tool_name'], call['arguments'])
        else:
            # Single search call or no batch support - process sequentially
            for idx, call in zip(search_indices, search_calls):
                results[idx] = self.base_handler.execute_tool(call['tool_name'], call['arguments'])
        
        # Process other tool types sequentially
        for idx, call in other_calls:
            logger.info(f"Executing other tool calls: {call['tool_name']}")
            results[idx] = self.base_handler.execute_tool(call['tool_name'], call['arguments'])
        
        return results
    
    def _format_search_results(self, candidates: List[Dict]) -> List[Dict]:
        """Apply snippet processing to search results (reuse existing logic)."""
        # Copy the snippet processing logic from _dynamic_search
        if self.base_handler.snippet_max_tokens and self.base_handler.snippet_max_tokens > 0 and self.base_handler.tokenizer:
            for cand in candidates:
                text = cand.get("text", "")
                tokens = self.base_handler.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.base_handler.snippet_max_tokens,
                )
                if len(tokens) < self.base_handler.snippet_max_tokens:
                    cand["snippet"] = text
                else:
                    cand["snippet"] = self.base_handler.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            for cand in candidates:
                cand["snippet"] = cand.get("text", "")
        
        # Format results (copy from _dynamic_search)
        results = []
        for cand in candidates:
            result = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                result["score"] = cand["score"]
            if cand.get("url"):
                result["url"] = cand["url"]
            results.append(result)
        
        return results
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base handler."""
        return getattr(self.base_handler, name)


class ConfigurableToolHandler:
    """
    Thin wrapper that manages multiple SearchToolHandlers based on JSON configuration.
    Delegates actual tool execution to the vendor's SearchToolHandler.
    """
    
    def __init__(self, config: Dict = None, config_path: Path = None,
                 searcher_factory = None, k: int = 5, 
                 snippet_max_tokens: int = 512, include_get_document: bool = False):
        """
        Initialize with config from various sources.
        
        Args:
            config: Direct config dictionary
            config_path: Path to config JSON file 
            searcher_factory: Function that creates searchers (searcher_type) -> searcher
            k: Number of search results per query
            snippet_max_tokens: Max tokens per result snippet
            include_get_document: Whether to include get_document tool
        """
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.include_get_document = include_get_document
        
        # Load config from appropriate source
        if config:
            # Direct config dict provided
            self.config = config
            self.config_file = "provided_directly"
            self.config_name = config.get('config_name', 'unknown')
        elif config_path:
            # Load from specified path
            self.config_file = str(config_path)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            # Parse config name from standard tool-config and run output structures.
            from src.datasets.BrowseCompPlus.utils.path_utils import parse_config_name
            self.config_name = parse_config_name(Path(config_path))
        else:
            raise ValueError("Must provide either config or config_path")
        
        logger.info(f"Loaded config '{self.config_name}': {self.config['description']}")
        
        # Create shared SearchToolHandlers for each searcher type
        self.handlers = {}
        self.tool_mapping = {}
        
        for tool in self.config['tools']:
            tool_id = tool['tool_id']
            tool_name = tool['tool_definition']['name']
            searcher_config = tool['searcher_config']
            searcher_type = searcher_config['searcher_type']
            # BrowseCompPlus invariant: every tool must declare parameters.
            self._validate_tool_parameters(tool)
            
            # Create shared handler if we haven't seen this searcher type before
            if searcher_type not in self.handlers:
                searcher = searcher_factory(searcher_type)
                # Check if any tool of this searcher type needs get_document
                needs_get_doc = self.include_get_document and any(
                    t['tool_id'] == 'get_document' and 
                    t['searcher_config']['searcher_type'] == searcher_type 
                    for t in self.config['tools']
                )
                base_handler = SearchToolHandler(
                    searcher, 
                    snippet_max_tokens=self.snippet_max_tokens, 
                    k=self.k,
                    include_get_document=needs_get_doc
                )
                # Wrap with dynamic filtering capability
                self.handlers[searcher_type] = DynamicFilteringSearchToolHandler(base_handler)
                logger.info(f"Created shared handler for {searcher_type}")
            
            # Map tool name to handler and filter configuration
            self.tool_mapping[tool_name] = {
                'handler': self.handlers[searcher_type],
                'tool_id': tool_id,
                'searcher_config': searcher_config,
                'param_mapping': self._get_param_mapping(tool)
            }

    def _validate_tool_parameters(self, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate BrowseCompPlus tool parameter schema and return properties dict."""
        tool_def = tool_config.get('tool_definition')
        if not isinstance(tool_def, dict):
            raise ValueError("Invalid tool config: missing 'tool_definition' object")

        tool_name = tool_def.get('name', '<unknown>')
        params = tool_def.get('parameters')
        if not isinstance(params, dict):
            raise ValueError(
                f"Tool '{tool_name}' is missing 'tool_definition.parameters'. "
                "BrowseCompPlus requires explicit parameter schemas."
            )

        properties = params.get('properties')
        if not isinstance(properties, dict) or not properties:
            raise ValueError(
                f"Tool '{tool_name}' has empty/missing parameter properties. "
                "BrowseCompPlus requires at least one parameter per tool."
            )

        return properties
    
    def _get_param_mapping(self, tool_config: Dict[str, Any]) -> Dict[str, str]:
        """Get parameter name mappings for opaque tools."""
        # Map from opaque parameter names to standard handler names.
        params = self._validate_tool_parameters(tool_config)
        tool_id = tool_config.get('tool_id', '')
        tool_name = tool_config.get('tool_definition', {}).get('name', '<unknown>')

        if tool_id.startswith('search_'):
            if 'query' in params:
                return {}  # Standard name
            if 'input' in params:
                return {'input': 'query'}
            raise ValueError(
                f"Search tool '{tool_name}' must expose parameter 'query' or 'input'; "
                f"found {list(params.keys())}"
            )

        if tool_id == 'get_document':
            if 'docid' in params:
                return {}  # Standard name
            if 'id' in params:
                return {'id': 'docid'}
            raise ValueError(
                f"get_document tool '{tool_name}' must expose parameter 'docid' or 'id'; "
                f"found {list(params.keys())}"
            )

        raise ValueError(f"Unknown tool type for '{tool_name}': {tool_id}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions from config."""
        tools = []
        
        for tool_config in self.config['tools']:
            tool_def = tool_config['tool_definition'].copy()
            
            # Replace only the {k} token. Do not run full .format(...) here:
            # optimized descriptions may include literal JSON examples like {"query": "..."}.
            if 'description' in tool_def:
                tool_def['description'] = tool_def['description'].replace("{k}", str(self.k))
            
            # Set the function type
            tool_def['type'] = 'function'
            
            # Ensure strict mode for OpenAI
            tool_def['strict'] = True
            
            tools.append(tool_def)
        
        return tools
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call by delegating to the appropriate SearchToolHandler with dynamic filtering."""
        if tool_name not in self.tool_mapping:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        mapping = self.tool_mapping[tool_name]
        handler = mapping['handler']
        tool_id = mapping['tool_id']
        searcher_config = mapping['searcher_config']
        
        # Remap opaque parameter names to standard ones
        remapped_args = {}
        for opaque_name, value in arguments.items():
            standard_name = mapping['param_mapping'].get(opaque_name, opaque_name)
            remapped_args[standard_name] = value
        
        # Add dynamic filter parameters from the tool's searcher config
        filter_category = searcher_config.get('filter_category')
        filter_domain = searcher_config.get('filter_domain')
        
        if filter_category and filter_category != 'none':
            remapped_args['filter_category'] = filter_category
        if filter_domain:
            remapped_args['filter_domain'] = filter_domain
        
        # Determine the actual tool name for the handler
        if tool_id.startswith('search_'):
            actual_tool = 'search'
        elif tool_id == 'get_document':
            actual_tool = 'get_document'
        else:
            raise ValueError(f"Unknown tool type: {tool_id}")
        
        # Delegate to the vendor's SearchToolHandler with dynamic filtering
        return handler.execute_tool(actual_tool, remapped_args)
    
    def execute_batch_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Execute multiple tool calls, delegating to appropriate handlers with batch optimization."""
        
        # Group tool calls by handler to enable batch processing within each handler
        handler_groups = {}
        tool_call_indices = {}
        
        for i, call in enumerate(tool_calls):
            tool_name = call['tool_name']
            if tool_name not in self.tool_mapping:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            mapping = self.tool_mapping[tool_name]
            handler = mapping['handler']
            
            # Group by handler
            if handler not in handler_groups:
                handler_groups[handler] = []
                tool_call_indices[handler] = []
            
            # Remap arguments and add filter parameters (same as execute_tool)
            arguments = call['arguments']
            searcher_config = mapping['searcher_config']
            
            remapped_args = {}
            for opaque_name, value in arguments.items():
                standard_name = mapping['param_mapping'].get(opaque_name, opaque_name)
                remapped_args[standard_name] = value
            
            # Add dynamic filter parameters
            filter_category = searcher_config.get('filter_category')
            filter_domain = searcher_config.get('filter_domain')
            
            if filter_category and filter_category != 'none':
                remapped_args['filter_category'] = filter_category
            if filter_domain:
                remapped_args['filter_domain'] = filter_domain
            
            # Determine actual tool name
            tool_id = mapping['tool_id']
            if tool_id.startswith('search_'):
                actual_tool = 'search'
            elif tool_id == 'get_document':
                actual_tool = 'get_document'
            else:
                raise ValueError(f"Unknown tool type: {tool_id}")
            
            # Add to handler group
            handler_groups[handler].append({
                'tool_name': actual_tool,
                'arguments': remapped_args
            })
            tool_call_indices[handler].append(i)
        
        # Execute each handler group and collect results
        results = [None] * len(tool_calls)
        
        for handler, group_calls in handler_groups.items():
            group_indices = tool_call_indices[handler]
            
            # Use batch execution - all handlers are DynamicFilteringSearchToolHandler with batch support
            group_results = handler.execute_batch_tools(group_calls)
            
            # Map results back to original positions
            for idx, result in zip(group_indices, group_results):
                results[idx] = result
        
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "config_name": self.config_name,
            "config_description": self.config.get('description', 'No description'),
            "config_file": str(self.config_file),
            "num_tools": len(self.config['tools']),
            "tool_names": [tool['tool_definition']['name'] for tool in self.config['tools']],
            "num_shared_handlers": len(self.handlers),
            "handler_types": list(self.handlers.keys()),
            "k": self.k,
            "snippet_max_tokens": self.snippet_max_tokens
        }
