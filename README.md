# MCP Servers

This repository contains a collection of Model Context Protocol (MCP) servers for enhancing AI assistants with custom capabilities. Each subdirectory contains a self-contained MCP implementation for a specific use case.

## What are MCPs?

Model Context Protocols (MCPs) allow AI assistants to access external tools, data sources, and capabilities beyond their base knowledge. MCPs enable AI models to:

- Access up-to-date information from external sources
- Perform specialized tasks and calculations
- Access proprietary or private data
- Interface with specific tools and systems

## Available MCPs

### Great Learning MCP

Located in: `great_learning/`

The Great Learning MCP enables AI assistants to access and search information about Great Learning's educational courses and programs. Features include:

- Semantic search across Great Learning content
- Detailed information about specific courses
- Course comparison capabilities
- Full access to Great Learning documentation

[See Great Learning MCP README](great_learning/README.md)

## Adding an MCP to Your AI Assistant

To use these MCPs with your AI assistant, add the configuration to your assistant's MCP configuration file:

### Supported Platforms

- **Cursor**: `~/.cursor/mcp.json`
- **Windsurf**: `~/.codeium/windsurf/mcp_config.json`
- **Claude Desktop**: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- **Claude Code**: `~/.claude.json`

### Configuration Format

```json
{
    "mcpServers": {
        "great-learning-mcp": {
            "command": "/path/to/python/virtualenv",
            "args": [
                "/path/to/mcp-servers/great_learning/great_learning_mcp.py"
            ]
        }
    }
}
```

## Requirements

- Python 3.12+
- Poetry for dependency management
- Required dependencies per specific MCP

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mcp-servers.git
   cd mcp-servers
   ```

2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Install specific requirements for each MCP (see individual READMEs)

## Development

To add a new MCP to this repository:

1. Create a new directory for your MCP
2. Implement your MCP using the MCP SDK
3. Add a README.md file with usage instructions
4. Add a mcp_config.json file with configuration details
5. Add any additional required files

