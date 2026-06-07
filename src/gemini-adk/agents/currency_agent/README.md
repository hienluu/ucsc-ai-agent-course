## Current Agent Example

* Based on the one from [adk-samples](https://github.com/google/adk-samples/tree/main/python/agents/currency-agent)
* Use the latest API updates from [Frankfurter website](https://frankfurter.dev/)
    * Example: https://api.frankfurter.dev/v2/rate/USD/VND
* Use Doubleworld open source hosted model


### Local Deployment

In a terminal, start the MCP server from currency_agent folder
```
uv run mcp_server.py
```

Run the currency agent from agents folder
```
adk run currency_agent
```

or 

```
adk web .
```