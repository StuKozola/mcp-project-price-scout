import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import re
import ast

from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import MessageParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        # implement the logic to open and read the file_path
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)

                # check if mcpServers is in the config
                if "mcpServers" not in config:
                    raise ValueError(
                        "Configuration file is missing 'mcpServers' field.")
                return config
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which(
            "npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError(
                "The command must be a valid string and cannot be None.")

        # Fill it using the self.config dictionary
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]
                 } if self.config.get("env") else None,
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        # Check if the session exists
        if not self.session:
            raise RuntimeError(
                f"Server {self.name} is not initialized. Cannot list tools.")
        # call the session to get tools
        tools_response = await self.session.list_tools()
        results = []
        # Format the response. The starter code provides the loop; you just need to populate the tool_def dictionary
        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema
            }
            results.append(tool_def)
        return results

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        # ipmlement the retry loop
        if not self.session:
            raise RuntimeError(
                f"Server {self.name} is not initialized. Cannot execute tool.")
        try:
            logging.info(f"Executing {tool_name}...")
            result = await self.session.call_tool(name=tool_name, arguments=arguments, read_timeout_seconds=timedelta(seconds=60))
            return result
        except Exception as e:
            if retries > 0:
                logging.warning(
                    f"Tool {tool_name} failed with error: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                return await self.execute_tool(tool_name, arguments, retries - 1, delay * 2)
            else:
                logging.error(
                    f"Tool {tool_name} failed after retries. Error: {e}")
                raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(
                    f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client

    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:

            await self.sqlite_server.execute_tool("write_query", {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            })

            logging.info("✓ Data extraction tables initialized")

        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            # Validate prompt is not empty
            if not prompt or not prompt.strip():
                logging.error(
                    "Empty prompt provided to _get_structured_extraction")
                return '{"error": "empty prompt"}'

            response = self.anthropic.messages.create(
                max_tokens=15000,
                model='claude-sonnet-4-5-20250929',
                messages=[{'role': 'user', 'content': prompt}]
            )

            text_content = ""
            for content in response.content:
                if content.type == 'text':
                    text_content += content.text

            return text_content.strip()

        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    async def extract_and_store_data(self, user_query: str, llm_response: str,
                                     source_url: str = "") -> None:
        """Extract structured data from LLM response and store it."""
        try:
            # Skip if content is empty
            if not llm_response or not llm_response.strip():
                logger.warning("Skipping extraction - empty content")
                return

            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format.
            
            Text: {llm_response}
            
            Extract pricing plans with this EXACT structure. Use null for missing numeric values:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": 0.001,
                        "output_tokens": 0.002,
                        "currency": "USD",
                        "billing_period": "per-million-tokens",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations"
                    }}
                ]
            }}
            
            CRITICAL RULES:
            - Return ONLY valid JSON, no explanations or markdown
            - Use null (not "null" string) for missing numeric values
            - Escape all special characters in strings (quotes, backslashes, newlines)
            - Keep feature descriptions SHORT (max 50 chars each)
            - Do NOT include any text before or after the JSON
            - Do NOT wrap in code blocks or backticks
            - Pricing should be per million tokens if mentioned
            - Use simple, clean strings without special characters
            """

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            # logger.info(f"Attempting to parse extraction response: {extraction_response}")
            # Clean up the response - remove code blocks and extra whitespace
            extraction_response = extraction_response.strip()
            # Remove markdown code blocks (various formats)
            extraction_response = extraction_response.replace(
                "```json\n", "").replace("```json", "")
            extraction_response = extraction_response.replace(
                "```\n", "").replace("```", "")
            extraction_response = extraction_response.strip()

            # Log what we're trying to parse for debugging
            # logger.info(
            # f"Attempting to parse extraction response (first 500 chars): {extraction_response[:500]}")

            # Function to try fixing common JSON issues
            def try_fix_json(json_str: str) -> str:
                """Attempt to fix common JSON formatting issues"""
                # Replace problematic characters
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                # Fix multiple spaces
                json_str = ' '.join(json_str.split())
                return json_str

            try:
                pricing_data = json.loads(extraction_response)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Initial JSON parse failed: {e}. Trying fixes...")

                # Try to fix and parse again
                try:
                    fixed_response = try_fix_json(extraction_response)
                    pricing_data = json.loads(fixed_response)
                    logger.info("Successfully parsed after fixing")
                except json.JSONDecodeError:
                    # Try to extract JSON object from response
                    logger.warning("Trying to extract JSON object...")
                    start = extraction_response.find('{')
                    end = extraction_response.rfind('}') + 1
                    if start != -1 and end > start:
                        try:
                            extracted = extraction_response[start:end]
                            pricing_data = json.loads(try_fix_json(extracted))
                            logger.info(
                                "Successfully extracted and parsed JSON")
                        except json.JSONDecodeError as final_error:
                            logger.error(
                                f"All parsing attempts failed: {final_error}")
                            logger.error(
                                f"Full response (first 1000 chars): {extraction_response[:1000]}")
                            logger.error(
                                "Skipping data storage.")
                            return
                    else:
                        logger.error("No JSON object found in response")
                        logger.error(f"Full response: {extraction_response}")
                        return

            # Helper function to escape SQL strings
            def escape_sql(value: Any) -> str:
                """Escape single quotes for SQL by replacing ' with ''"""
                if value is None:
                    return "NULL"
                return str(value).replace("'", "''")

            for plan in pricing_data.get("plans", []):
                company_name = escape_sql(
                    pricing_data.get("company_name", "Unknown"))
                plan_name = escape_sql(plan.get("plan_name", "Unknown Plan"))

                # Handle numeric fields - use NULL for None values
                input_tokens = plan.get("input_tokens")
                input_tokens_sql = "NULL" if input_tokens is None else str(
                    input_tokens)

                output_tokens = plan.get("output_tokens")
                output_tokens_sql = "NULL" if output_tokens is None else str(
                    output_tokens)

                currency = escape_sql(plan.get("currency", "USD"))

                # Handle billing_period - use NULL for None
                billing_period = plan.get("billing_period")
                billing_period_sql = "NULL" if billing_period is None else f"'{escape_sql(billing_period)}'"

                features = escape_sql(json.dumps(plan.get("features", [])))
                limitations = escape_sql(plan.get("limitations", ""))
                source_query = escape_sql(user_query)

                q = {
                    "query": f"""
                    INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
                    VALUES (
                        '{company_name}',
                        '{plan_name}',
                        {input_tokens_sql},
                        {output_tokens_sql},
                        '{currency}',
                        {billing_period_sql},
                        '{features}',
                        '{limitations}',
                        '{source_query}')
                    """
                }
                # logger.info(f"Query to execute: {q['query']}")
                await self.sqlite_server.execute_tool("write_query", q)

            logger.info(
                f"Stored {len(pricing_data.get('plans', []))} pricing plans")

        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(
            api_key=api_key)  # , base_url="https://claude.vocareum.com")
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def check_database_for_query(self, query: str) -> str | None:
        """Check if the database has relevant information for the query."""
        if not self.sqlite_server:
            return None
        
        try:
            # Check if we have any data in the database
            result = await self.sqlite_server.execute_tool("read_query", {
                "query": "SELECT COUNT(*) as count FROM pricing_plans"
            })
            
            if result and result.content:
                count_text = result.content[0].text
                count = ast.literal_eval(count_text)[0]['count']
                
                if count > 0:
                    # Get all pricing data to check if it's relevant
                    pricing_result = await self.sqlite_server.execute_tool("read_query", {
                        "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans"
                    })
                    
                    if pricing_result and pricing_result.content:
                        data_text = pricing_result.content[0].text
                        logger.info(f"Found {count} pricing plans in database")
                        return data_text
            
            return None
        except Exception as e:
            logger.warning(f"Error checking database: {e}")
            return None

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        # Check if database has relevant information
        db_data = await self.check_database_for_query(query)
        
        # Prepare tools - exclude scraping tools if we have database data
        tools_to_use = self.available_tools
        if db_data:
            logger.info("Database has existing data, excluding scraping tools from available tools")
            # Filter out scraping tools
            scraping_tool_names = ['scrape_websites', 'fetch_url', 'scrape_url']
            tools_to_use = [tool for tool in self.available_tools if tool['name'] not in scraping_tool_names]
            
            # Add database context to the query
            enhanced_query = f"{query}\n\nAvailable pricing data from database:\n{db_data}"
            messages: List[MessageParam] = [{'role': 'user', 'content': enhanced_query}]
        else:
            logger.info("No relevant data in database, including scraping tools")
            messages: List[MessageParam] = [{'role': 'user', 'content': query}]
        
        response = self.anthropic.messages.create(
            max_tokens=15000,
            model='claude-sonnet-4-5-20250929',
            tools=tools_to_use,
            messages=messages
        )

        full_response = ""
        source_url = None
        scraped_contents = []  # Store all scraped content

        while True:
            # Collect assistant content
            assistant_content = []
            tool_uses = []

            for content in response.content:
                if content.type == 'text':
                    full_response += content.text + "\n"
                    print(content.text)  # Print response as we get it
                    assistant_content.append(content)
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    tool_uses.append(content)

            # Add assistant message to conversation
            messages.append({
                'role': 'assistant',
                'content': assistant_content
            })

            # If no tool uses, we're done
            if not tool_uses:
                break

            # Execute tools and collect results
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_args = tool_use.input
                tool_id = tool_use.id

                if tool_name in self.tool_to_server:
                    logging.info(
                        f"Tool {tool_name} requested with args {tool_args}")
                    server_name = self.tool_to_server[tool_name]

                    for server in self.servers:
                        if server.name == server_name:
                            tool_result = await server.execute_tool(tool_name, tool_args)
                            logging.info(f"Tool {tool_name} executed")

                            # Store scraped content for data extraction
                            if tool_name in ['fetch_url', 'scrape_url', 'read_file', 'scrape_websites']:
                                for content_item in tool_result.content:
                                    if hasattr(content_item, 'text'):
                                        scraped_contents.append({
                                            'tool': tool_name,
                                            'args': tool_args,
                                            'content': content_item.text
                                        })

                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': tool_id,
                                'content': tool_result.content
                            })
                            break
                else:
                    logger.warning(
                        f"Tool {tool_name} not found in available tools")

            # Add tool results to conversation
            if tool_results:
                messages.append({
                    'role': 'user',
                    'content': tool_results
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    max_tokens=15000,
                    model='claude-sonnet-4-5-20250929',
                    tools=self.available_tools,
                    messages=messages
                )
            else:
                break

        # Extract data from each scraped content
        if self.data_extractor and scraped_contents:
            for scraped in scraped_contents:
                await self.data_extractor.extract_and_store_data(
                    query,
                    scraped['content'],
                    scraped['args'].get('url', '') if isinstance(
                        scraped['args'], dict) else ''
                )
        # If no scraped contents but scrape_websites was called, check for existing files
        elif self.data_extractor and 'scrape' in query.lower():
            logger.info("Checking for existing scraped content files...")
            scraped_dir = Path(__file__).parent.parent / "scraped_content"
            if scraped_dir.exists():
                # Look for markdown files
                for markdown_file in scraped_dir.glob("*_markdown.txt"):
                    try:
                        with open(markdown_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Extract company name and URL from metadata
                        metadata_file = scraped_dir / "scraped_metadata.json"
                        url = ""
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                # Find matching provider
                                provider_name = markdown_file.stem.replace(
                                    '_markdown', '')
                                for provider, data in metadata.items():
                                    if provider.lower() == provider_name.lower():
                                        url = data.get('url', '')
                                        break

                        logger.info(
                            f"Processing existing scraped file: {markdown_file.name}")
                        # Only process if content is not empty
                        if content and content.strip():
                            await self.data_extractor.extract_and_store_data(
                                query,
                                content,
                                url
                            )
                        else:
                            logger.warning(
                                f"Skipping empty file: {markdown_file.name}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to process {markdown_file}: {e}")

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                # Skip empty queries
                if not query:
                    continue
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data(limit=5)
                    continue
                elif query.lower() == 'show data all':
                    await self.show_stored_data(limit=None)
                    continue

                await self.process_query(query)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self, limit: int = 5) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return

        try:
            # read from the pricing_plans table using the tool
            if limit is None:
                q = "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC"
            else:
                q = f"SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT {limit}"
            pricing = await self.sqlite_server.execute_tool("read_query", {
                "query": q
            })

            print("\nRecently Stored Data:")
            print("=" * 50)

            print("\nPricing Plans:")

            # Check if we have valid content
            if not pricing or not pricing.content or len(pricing.content) == 0:
                print("  No data available")
            else:
                result_text = pricing.content[0].text
                if not result_text or result_text.strip() == "":
                    print("  No data available")
                else:
                    # Parse the result
                    try:
                        plans = ast.literal_eval(result_text)
                        if not plans or len(plans) == 0:
                            print("  No data available")
                        else:
                            for plan in plans:
                                print(
                                    f"  • {plan['company_name']}: {plan['plan_name']} - Input Token ${plan['input_tokens']}, Output Tokens ${plan['output_tokens']}")
                    except (ValueError, SyntaxError) as parse_error:
                        print(f"  Error parsing data: {parse_error}")
                        print(f"  Raw result: {result_text}")

            print("=" * 50)
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(
                f"Available tools: {[tool['name'] for tool in self.available_tools]}")

            if self.sqlite_server:
                self.data_extractor = DataExtractor(
                    self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()

    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"

    server_config = config.load_config(config_file)

    servers = [Server(name, srv_config)
               for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
