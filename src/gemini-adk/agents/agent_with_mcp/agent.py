from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
import os

"""
Example of an agent that uses the Model Control Protocol (MCP) to call tools based on model instructions.
https://adk.dev/tools-custom/mcp-tools/
"""

TARGET_FOLDER_PATH = "./mcp_files"  # Update this path to a valid absolute path on your system

### define MCP as tools
fs_mcp_tool = McpToolset(connection_params=StdioConnectionParams(
        server_params = StdioServerParameters(
            command='npx', 
            args=["-y",  # Argument for npx to auto-confirm install 
                "@modelcontextprotocol/server-filesystem", 
                 # IMPORTANT: This MUST be an ABSOLUTE path to a folder the # npx process can access. # Replace with a valid absolute path on your system. # For example: "/Users/youruser/accessible_mcp_files" # or use a dynamically constructed absolute path: 
                os.path.abspath(TARGET_FOLDER_PATH)
             ],
        ), 
    )
)

# time and time covnersion tool example
time_mcp_tool = McpToolset(connection_params=StdioConnectionParams(
        server_params = StdioServerParameters(
            command='uvx', 
            args=["mcp-server-time"],
        ), 
    )
)

# sequential thinking MCP tools
thinking_mcp_tool = McpToolset(connection_params=StdioConnectionParams(
        server_params = StdioServerParameters(
            command='npx', 
            args=["-y",  # Argument for npx to auto-confirm install 
                "@modelcontextprotocol/server-sequential-thinking",            
             ],
        ), 
    )
)


root_agent = LlmAgent(
    model='gemini-3.1-flash-lite',
    name='mcp_example_agent',
    description='A helpful assistant for user questions.  You have access to tools that can help you answer user questions. Use them when appropriate.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[fs_mcp_tool, time_mcp_tool, thinking_mcp_tool]
)
