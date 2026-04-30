#!/usr/bin/env python3
"""Manual smoke test for BrowseCompPlus generation folder naming."""

from src.datasets.BrowseCompPlus.utils.path_utils import create_generation_dirname


class MockArgs:
    def __init__(self, **kwargs):
        self.model = "gpt-5"
        self.temperature = 1.0
        self.top_p = 1.0
        self.tool_choice = "required"
        self.max_tokens = 10000
        self.k = 5
        self.snippet_max_tokens = 512
        self.include_get_document = False
        self.hide_urls = False
        self.max_iterations = 100
        self.num_queries = 100
        self._resolved_prompt_key = "runtime_search_only"
        self.reasoning_effort = "medium"

        for key, value in kwargs.items():
            setattr(self, key, value)


def main():
    print("BrowseCompPlus generation dirname examples:\n")

    args = MockArgs()
    print("Default:")
    print("  ", create_generation_dirname(args))

    args = MockArgs(model="gpt-4o-2024-08-06", temperature=0.7, top_p=0.95)
    print("\nNon-reasoning model:")
    print("  ", create_generation_dirname(args))

    args = MockArgs(include_get_document=True, _resolved_prompt_key="runtime_with_get_document")
    print("\nWith get_document:")
    print("  ", create_generation_dirname(args))


if __name__ == "__main__":
    main()
