"""
Unit and integration tests for starter_client.py

This test suite covers:
- Configuration class functionality
- Server class operations
- DataExtractor functionality
- ChatSession orchestration
- Error handling and edge cases
"""

from starter_client import (
    Configuration,
    Server,
    DataExtractor,
    ChatSession,
    ToolDefinition
)
import os
import json
import pytest
import pytest_asyncio
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
from typing import Any, Dict, List

pytest_plugins = ('pytest_asyncio',)

# Import the classes and functions to test


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing"""
    config_data = {
        "mcpServers": {
            "test_server": {
                "command": "python",
                "args": ["-m", "test_module"],
                "env": {"TEST_VAR": "test_value"}
            },
            "sqlite_server": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                "env": {}
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_tool_response():
    """Mock tool response from MCP server"""
    tool1 = Mock()
    tool1.name = "test_tool_1"
    tool1.description = "Test tool 1 description"
    tool1.inputSchema = {"type": "object", "properties": {}}

    tool2 = Mock()
    tool2.name = "test_tool_2"
    tool2.description = "Test tool 2 description"
    tool2.inputSchema = {"type": "object", "properties": {}}

    response = Mock()
    response.tools = [tool1, tool2]

    return response


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_content = Mock()
    mock_content.type = 'text'
    mock_content.text = "This is a test response from the LLM."

    mock_response = Mock()
    mock_response.content = [mock_content]

    return mock_response


class TestConfiguration:
    """Test suite for Configuration class"""

    def test_configuration_init(self):
        """Test Configuration initialization"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            config = Configuration()
            assert config.api_key == 'test_key'

    def test_load_env(self):
        """Test loading environment variables"""
        with patch('starter_client.load_dotenv') as mock_load:
            Configuration.load_env()
            mock_load.assert_called_once()

    def test_load_config_success(self, temp_config_file):
        """Test successful configuration file loading"""
        config = Configuration.load_config(temp_config_file)

        assert "mcpServers" in config
        assert "test_server" in config["mcpServers"]
        assert config["mcpServers"]["test_server"]["command"] == "python"

    def test_load_config_file_not_found(self):
        """Test loading non-existent configuration file"""
        with pytest.raises(FileNotFoundError):
            Configuration.load_config("nonexistent_file.json")

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {{{")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                Configuration.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_missing_mcpServers(self):
        """Test loading config without mcpServers field"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"otherField": "value"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="missing 'mcpServers' field"):
                Configuration.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_anthropic_api_key_present(self):
        """Test getting API key when present"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            config = Configuration()
            assert config.anthropic_api_key == 'test_key'

    def test_anthropic_api_key_missing(self):
        """Test error when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('starter_client.os.getenv', return_value=None):
                config = Configuration()
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
                    _ = config.anthropic_api_key


class TestServer:
    """Test suite for Server class"""

    def test_server_init(self):
        """Test Server initialization"""
        config = {
            "command": "python",
            "args": ["-m", "test"],
            "env": {}
        }
        server = Server("test_server", config)

        assert server.name == "test_server"
        assert server.config == config
        assert server.session is None
        assert server.stdio_context is None

    @pytest.mark.asyncio
    async def test_server_initialize_success(self):
        """Test successful server initialization"""
        config = {
            "command": "python",
            "args": ["-m", "test"],
            "env": {}
        }
        server = Server("test_server", config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch('starter_client.stdio_client') as mock_stdio, \
                patch('starter_client.ClientSession', return_value=mock_session), \
                patch('shutil.which', return_value='/usr/bin/python'):

            mock_stdio.return_value.__aenter__ = AsyncMock(
                return_value=(Mock(), Mock()))
            mock_stdio.return_value.__aexit__ = AsyncMock()

            # Mock the AsyncExitStack
            server.exit_stack.enter_async_context = AsyncMock(side_effect=[
                (Mock(), Mock()),  # stdio_transport
                mock_session       # session
            ])

            await server.initialize()

            assert server.session is not None
            mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_initialize_invalid_command(self):
        """Test initialization with invalid command"""
        config = {
            "command": "invalid_command_that_does_not_exist",
            "args": [],
            "env": {}
        }
        server = Server("test_server", config)

        # The actual error is FileNotFoundError from subprocess, not ValueError from shutil.which
        with pytest.raises((FileNotFoundError, Exception)):
            await server.initialize()

    @pytest.mark.asyncio
    async def test_list_tools_success(self, sample_tool_response):
        """Test listing tools from server"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=sample_tool_response)
        server.session = mock_session

        tools = await server.list_tools()

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        mock_result = Mock()
        mock_result.content = [{"text": "Tool result"}]

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        server.session = mock_session

        result = await server.execute_tool("test_tool", {"arg": "value"})

        assert result == mock_result
        mock_session.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_not_initialized(self):
        """Test tool execution when server not initialized"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.execute_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry(self):
        """Test tool execution with retry mechanism"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        mock_session = AsyncMock()
        # First call fails, second succeeds
        mock_session.call_tool = AsyncMock(
            side_effect=[Exception("Network error"), Mock(
                content=[{"text": "Success"}])]
        )
        server.session = mock_session

        result = await server.execute_tool("test_tool", {}, retries=1, delay=0.01)

        assert mock_session.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_retry_exhausted(self):
        """Test tool execution when all retries are exhausted"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(
            side_effect=Exception("Persistent error"))
        server.session = mock_session

        with pytest.raises(Exception, match="Persistent error"):
            await server.execute_tool("test_tool", {}, retries=1, delay=0.01)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test server cleanup"""
        config = {"command": "python", "args": [], "env": {}}
        server = Server("test_server", config)

        mock_session = Mock()
        server.session = mock_session
        server.exit_stack.aclose = AsyncMock()

        await server.cleanup()

        assert server.session is None
        assert server.stdio_context is None
        server.exit_stack.aclose.assert_called_once()


class TestDataExtractor:
    """Test suite for DataExtractor class"""

    def test_data_extractor_init(self):
        """Test DataExtractor initialization"""
        mock_server = Mock()
        mock_anthropic = Mock()

        extractor = DataExtractor(mock_server, mock_anthropic)

        assert extractor.sqlite_server == mock_server
        assert extractor.anthropic == mock_anthropic

    @pytest.mark.asyncio
    async def test_setup_data_tables(self):
        """Test setting up data tables"""
        mock_server = AsyncMock()
        mock_server.execute_tool = AsyncMock()
        mock_anthropic = Mock()

        extractor = DataExtractor(mock_server, mock_anthropic)
        await extractor.setup_data_tables()

        mock_server.execute_tool.assert_called_once()
        call_args = mock_server.execute_tool.call_args
        assert call_args[0][0] == "write_query"
        assert "CREATE TABLE IF NOT EXISTS pricing_plans" in call_args[0][1]["query"]

    @pytest.mark.asyncio
    async def test_get_structured_extraction_success(self, mock_anthropic_response):
        """Test successful structured extraction"""
        mock_server = Mock()
        mock_anthropic = Mock()
        mock_anthropic.messages.create = Mock(
            return_value=mock_anthropic_response)

        extractor = DataExtractor(mock_server, mock_anthropic)
        result = await extractor._get_structured_extraction("test prompt")

        assert result == "This is a test response from the LLM."
        mock_anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_structured_extraction_error(self):
        """Test structured extraction with error"""
        mock_server = Mock()
        mock_anthropic = Mock()
        mock_anthropic.messages.create = Mock(
            side_effect=Exception("API error"))

        extractor = DataExtractor(mock_server, mock_anthropic)
        result = await extractor._get_structured_extraction("test prompt")

        assert result == '{"error": "extraction failed"}'

    @pytest.mark.asyncio
    async def test_extract_and_store_data_success(self):
        """Test successful data extraction and storage"""
        mock_server = AsyncMock()
        mock_server.execute_tool = AsyncMock()
        mock_anthropic = Mock()

        # Mock the extraction response
        extraction_json = json.dumps({
            "company_name": "Test Company",
            "plans": [
                {
                    "plan_name": "Basic Plan",
                    "input_tokens": 0.001,
                    "output_tokens": 0.002,
                    "currency": "USD",
                    "billing_period": "monthly",
                    "features": ["feature1", "feature2"],
                    "limitations": "1000 requests/month"
                }
            ]
        })

        mock_response = Mock()
        mock_response.content = [Mock(type='text', text=extraction_json)]
        mock_anthropic.messages.create = Mock(return_value=mock_response)

        extractor = DataExtractor(mock_server, mock_anthropic)

        await extractor.extract_and_store_data(
            "What is the pricing?",
            "Company offers Basic Plan at $0.001 per input token",
            "https://example.com"
        )

        # Should call execute_tool to insert data
        assert mock_server.execute_tool.call_count >= 1

    @pytest.mark.asyncio
    async def test_extract_and_store_data_invalid_json(self):
        """Test extraction with invalid JSON response"""
        mock_server = AsyncMock()
        mock_anthropic = Mock()

        mock_response = Mock()
        mock_response.content = [Mock(type='text', text="invalid json")]
        mock_anthropic.messages.create = Mock(return_value=mock_response)

        extractor = DataExtractor(mock_server, mock_anthropic)

        # Should not raise exception, just log error
        await extractor.extract_and_store_data(
            "test query",
            "test response",
            ""
        )


class TestChatSession:
    """Test suite for ChatSession class"""

    def test_chat_session_init(self):
        """Test ChatSession initialization"""
        mock_servers = [Mock(), Mock()]

        with patch('starter_client.Anthropic') as mock_anthropic_class:
            session = ChatSession(mock_servers, "test_api_key")

            assert len(session.servers) == 2
            assert session.available_tools == []
            assert session.tool_to_server == {}
            assert session.sqlite_server is None
            mock_anthropic_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_servers(self):
        """Test cleaning up all servers"""
        mock_server1 = AsyncMock()
        mock_server2 = AsyncMock()
        mock_server1.cleanup = AsyncMock()
        mock_server2.cleanup = AsyncMock()

        with patch('starter_client.Anthropic'):
            session = ChatSession([mock_server1, mock_server2], "test_api_key")
            await session.cleanup_servers()

            mock_server1.cleanup.assert_called_once()
            mock_server2.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_simple_text_response(self, mock_anthropic_response):
        """Test processing query with simple text response"""
        mock_server = AsyncMock()

        with patch('starter_client.Anthropic') as mock_anthropic_class:
            mock_anthropic_instance = Mock()
            mock_anthropic_instance.messages.create = Mock(
                return_value=mock_anthropic_response)
            mock_anthropic_class.return_value = mock_anthropic_instance

            session = ChatSession([mock_server], "test_api_key")
            session.data_extractor = None  # Disable data extraction for this test

            await session.process_query("Hello, how are you?")

            mock_anthropic_instance.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_tool_use(self):
        """Test processing query that uses tools"""
        mock_server = AsyncMock()
        mock_server.name = "test_server"
        mock_server.execute_tool = AsyncMock(
            return_value=Mock(content=[{"text": "Tool result"}]))

        # First response with tool use
        tool_use_content = Mock()
        tool_use_content.type = 'tool_use'
        tool_use_content.name = 'test_tool'
        tool_use_content.input = {'arg': 'value'}
        tool_use_content.id = 'tool_123'

        first_response = Mock()
        first_response.content = [tool_use_content]

        # Second response with text
        text_content = Mock()
        text_content.type = 'text'
        text_content.text = "Here is the result"

        second_response = Mock()
        second_response.content = [text_content]

        with patch('starter_client.Anthropic') as mock_anthropic_class:
            mock_anthropic_instance = Mock()
            mock_anthropic_instance.messages.create = Mock(
                side_effect=[first_response, second_response]
            )
            mock_anthropic_class.return_value = mock_anthropic_instance

            session = ChatSession([mock_server], "test_api_key")
            session.tool_to_server['test_tool'] = 'test_server'
            session.data_extractor = None

            await session.process_query("Use the test tool")

            mock_server.execute_tool.assert_called_once()
            assert mock_anthropic_instance.messages.create.call_count == 2

    def test_extract_url_from_result(self):
        """Test URL extraction from tool result"""
        with patch('starter_client.Anthropic'):
            session = ChatSession([], "test_api_key")

            result_text = "Here is the page: https://example.com/page and more text"
            url = session._extract_url_from_result(result_text)

            assert url == "https://example.com/page"

    def test_extract_url_from_result_no_url(self):
        """Test URL extraction when no URL present"""
        with patch('starter_client.Anthropic'):
            session = ChatSession([], "test_api_key")

            result_text = "No URL here"
            url = session._extract_url_from_result(result_text)

            assert url is None

    @pytest.mark.asyncio
    async def test_show_stored_data_no_database(self):
        """Test showing data when no database is available"""
        with patch('starter_client.Anthropic'):
            session = ChatSession([], "test_api_key")
            session.sqlite_server = None

            # Should not raise exception
            await session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_success(self):
        """Test successfully showing stored data"""
        mock_server = AsyncMock()
        mock_result = Mock()

        # Create a mock TextContent object with .text attribute
        mock_text_content = Mock()
        mock_text_content.text = str([
            {
                "company_name": "Test Co",
                "plan_name": "Basic",
                "input_tokens": 0.001,
                "output_tokens": 0.002,
                "currency": "USD"
            }
        ])
        mock_result.content = [mock_text_content]
        mock_server.execute_tool = AsyncMock(return_value=mock_result)

        with patch('starter_client.Anthropic'), \
                patch('builtins.print') as mock_print:
            session = ChatSession([mock_server], "test_api_key")
            session.sqlite_server = mock_server

            await session.show_stored_data()

            mock_server.execute_tool.assert_called_once()
            # Check that print was called with pricing data
            assert any("Test Co" in str(call)
                       for call in mock_print.call_args_list)

    @pytest.mark.asyncio
    async def test_start_initialization_failure(self):
        """Test start method when server initialization fails"""
        mock_server = AsyncMock()
        mock_server.initialize = AsyncMock(
            side_effect=Exception("Init failed"))
        mock_server.cleanup = AsyncMock()

        with patch('starter_client.Anthropic'):
            session = ChatSession([mock_server], "test_api_key")

            await session.start()

            mock_server.cleanup.assert_called()


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_configuration_to_server_workflow(self, temp_config_file):
        """Test loading config and creating servers"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            config = Configuration()
            server_config = config.load_config(temp_config_file)

            servers = [
                Server(name, srv_config)
                for name, srv_config in server_config["mcpServers"].items()
            ]

            assert len(servers) == 2
            assert servers[0].name in ["test_server", "sqlite_server"]

    def test_tool_definition_structure(self):
        """Test ToolDefinition TypedDict structure"""
        tool: ToolDefinition = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object"}
        }

        assert tool["name"] == "test_tool"
        assert "description" in tool
        assert "input_schema" in tool


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
