#!/usr/bin/env python3
"""Test script to verify Fireworks.ai pricing scraping and database storage"""

from dotenv import load_dotenv
from starter_client import Configuration, Server, ChatSession
import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


load_dotenv()


async def test_fireworks_scraping():
    """Test the full workflow: scrape -> extract -> store -> display"""

    print("=" * 60)
    print("Testing Fireworks.ai Pricing Scrape and Database Storage")
    print("=" * 60)

    # Change to parent directory for proper paths
    original_dir = os.getcwd()
    parent_dir = Path(__file__).parent.parent
    os.chdir(parent_dir)

    try:
        # Load configuration
        config = Configuration()
        config_file = Path("mcp-project/server_config.json")
        server_config = config.load_config(config_file)

        # Create servers
        servers = [Server(name, srv_config)
                   for name, srv_config in server_config["mcpServers"].items()]

        # Create chat session
        chat_session = ChatSession(servers, config.anthropic_api_key)

        # Initialize servers
        print("\n1. Initializing servers...")
        for server in chat_session.servers:
            await server.initialize()
            if "sqlite" in server.name.lower():
                chat_session.sqlite_server = server

        # Collect tools
        print("\n2. Collecting available tools...")
        for server in chat_session.servers:
            tools = await server.list_tools()
            chat_session.available_tools.extend(tools)
            for tool in tools:
                chat_session.tool_to_server[tool["name"]] = server.name

        print(
            f"   Available tools: {[t['name'] for t in chat_session.available_tools]}")

        # Setup data extractor
        if chat_session.sqlite_server:
            from starter_client import DataExtractor
            chat_session.data_extractor = DataExtractor(
                chat_session.sqlite_server, chat_session.anthropic)
            await chat_session.data_extractor.setup_data_tables()
            print("   ✓ Data extraction enabled")

        # Test query
        print("\n3. Processing query...")
        query = "Scrape https://fireworks.ai/pricing and tell me about their pricing for LLM inference"
        print(f"   Query: {query}")
        print("\n   Response:")
        print("   " + "-" * 56)

        await chat_session.process_query(query)

        print("   " + "-" * 56)

        # Show stored data
        print("\n4. Checking stored data...")
        await chat_session.show_stored_data()

        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await chat_session.cleanup_servers()
        os.chdir(original_dir)
