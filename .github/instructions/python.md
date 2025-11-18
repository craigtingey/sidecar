---
applyTo: "**/*.py"
---

# Python Code Guidelines

## Code Structure

- Place imports at the top, grouped by: standard library, third-party, local
- Use blank lines to separate logical sections
- Keep functions short and focused (ideally under 50 lines)
- Use docstrings for functions and classes when their purpose isn't immediately obvious

## Type Hints

- Always use type hints for function parameters and return values
- Use `Optional[Type]` for optional parameters
- Use `Union[Type1, Type2]` or `Type1 | Type2` (Python 3.10+) for multiple types
- Use `None` as return type for functions that don't return a value

## Async Programming

- Use `async def` for functions that perform I/O operations
- Use `await` when calling async functions
- Use `async with` for async context managers (e.g., aiohttp sessions)
- Avoid blocking operations in async functions

## Error Handling

- Use try/except blocks for operations that might fail
- Be specific about the exceptions you catch
- Always clean up resources in finally blocks or use context managers
- Don't catch exceptions silently - log them or re-raise with context

## Resource Management

- Use context managers (`with` statements) for file operations
- Close HTTP sessions and file handles explicitly or use async context managers
- Clean up temporary files and directories before returning from functions
- Use `tempfile` module for creating temporary files/directories

## FastAPI Best Practices

- Use Pydantic models for request/response validation
- Use dependency injection for shared resources
- Return appropriate HTTP status codes
- Use HTTPException for error responses
- Leverage FastAPI's automatic OpenAPI documentation

## Security

- Never log sensitive data (API keys, secrets, credentials)
- Validate all user inputs
- Use environment variables for all configuration
- Sanitize error messages before sending to clients
