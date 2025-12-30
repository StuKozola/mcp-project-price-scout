"""
Manual test script for starter_server.py

This script provides interactive tests and examples for testing the MCP server manually.
Run this script to test the server functions directly without pytest.
"""

from starter_server import scrape_websites, extract_scraped_info, SCRAPE_DIR
import os
import json
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_scrape_single_website():
    """Test scraping a single website"""
    print_section("Test 1: Scrape Single Website")

    # Example websites - replace with real URLs if you have an API key
    websites = {
        "example_provider": "https://example.com"
    }

    print(f"Attempting to scrape: {websites}")
    print(f"Note: This requires a valid FIRECRAWL_API_KEY environment variable")
    print(f"or pass api_key parameter to the function.\n")

    try:
        # Uncomment the next line if you have a valid API key
        # result = scrape_websites(websites)
        # print(f"✓ Successfully scraped: {result}")

        print("⚠ Test skipped - requires valid API key")
        print("To run: Set FIRECRAWL_API_KEY environment variable and uncomment code")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_scrape_multiple_websites():
    """Test scraping multiple websites"""
    print_section("Test 2: Scrape Multiple Websites")

    websites = {
        "site1": "https://example1.com",
        "site2": "https://example2.com",
        "site3": "https://example3.com"
    }

    print(f"Attempting to scrape {len(websites)} websites:")
    for name, url in websites.items():
        print(f"  - {name}: {url}")

    print("\n⚠ Test skipped - requires valid API key")


def test_extract_existing_data():
    """Test extracting data from scraped content"""
    print_section("Test 3: Extract Scraped Information")

    # Check if metadata file exists
    metadata_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json")

    if not os.path.exists(metadata_path):
        print("⚠ No scraped data found.")
        print(f"Expected metadata file at: {metadata_path}")
        print("Run scrape_websites first to create test data.\n")
        return

    # Load and display available providers
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✓ Found {len(metadata)} scraped providers:\n")
        for provider_name, data in metadata.items():
            print(f"  Provider: {provider_name}")
            print(f"    URL: {data.get('url', 'N/A')}")
            print(f"    Domain: {data.get('domain', 'N/A')}")
            print(f"    Scraped at: {data.get('scraped_at', 'N/A')}")
            print(f"    Formats: {', '.join(data.get('formats', []))}")
            print()

        # Test extraction by different identifiers
        if metadata:
            first_provider = list(metadata.keys())[0]
            first_data = metadata[first_provider]

            print("\nTesting extraction methods:")

            # Extract by provider name
            print(f"\n1. Extract by provider name: '{first_provider}'")
            result = extract_scraped_info(first_provider)
            result_data = json.loads(result)
            if "error" in result_data:
                print(f"   ✗ Error: {result_data['error']}")
            else:
                print(
                    f"   ✓ Success! Retrieved data for: {result_data.get('provider_name')}")
                print(
                    f"   Content preview: {str(result_data.get('content', {}))[:100]}...")

            # Extract by URL
            url = first_data.get('url')
            if url:
                print(f"\n2. Extract by URL: '{url}'")
                result = extract_scraped_info(url)
                result_data = json.loads(result)
                if "error" in result_data:
                    print(f"   ✗ Error: {result_data['error']}")
                else:
                    print(
                        f"   ✓ Success! Retrieved data for: {result_data.get('provider_name')}")

            # Extract by domain
            domain = first_data.get('domain')
            if domain:
                print(f"\n3. Extract by domain: '{domain}'")
                result = extract_scraped_info(domain)
                result_data = json.loads(result)
                if "error" in result_data:
                    print(f"   ✗ Error: {result_data['error']}")
                else:
                    print(
                        f"   ✓ Success! Retrieved data for: {result_data.get('provider_name')}")

            # Test non-existent identifier
            print(f"\n4. Extract non-existent identifier: 'nonexistent'")
            result = extract_scraped_info("nonexistent")
            result_data = json.loads(result)
            if "error" in result_data:
                print(f"   ✓ Correctly handled: {result_data['error']}")
            else:
                print(f"   ✗ Should have returned error")

    except Exception as e:
        print(f"✗ Error reading metadata: {e}")


def test_metadata_structure():
    """Test and validate metadata structure"""
    print_section("Test 4: Validate Metadata Structure")

    metadata_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json")

    if not os.path.exists(metadata_path):
        print("⚠ No metadata file found")
        return

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✓ Successfully loaded metadata with {len(metadata)} entries\n")

        required_fields = [
            'provider_name', 'url', 'domain', 'scraped_at',
            'formats', 'success', 'content_files', 'title', 'description'
        ]

        for provider_name, data in metadata.items():
            print(f"Validating: {provider_name}")
            missing_fields = [
                field for field in required_fields if field not in data]

            if missing_fields:
                print(f"  ✗ Missing fields: {', '.join(missing_fields)}")
            else:
                print(f"  ✓ All required fields present")

            # Validate content files exist
            content_files = data.get('content_files', {})
            for format_type, file_path in content_files.items():
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"    ✓ {format_type} file exists ({size} bytes)")
                else:
                    print(f"    ✗ {format_type} file missing: {file_path}")

            print()

    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in metadata file: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_directory_structure():
    """Test and display directory structure"""
    print_section("Test 5: Directory Structure")

    print(f"Scrape directory: {SCRAPE_DIR}")

    if not os.path.exists(SCRAPE_DIR):
        print(f"⚠ Directory does not exist")
        print(f"It will be created when first scrape operation runs")
        return

    print(f"✓ Directory exists\n")

    # List all files
    files = list(Path(SCRAPE_DIR).iterdir())
    print(f"Found {len(files)} files/directories:\n")

    for item in sorted(files):
        if item.is_file():
            size = item.stat().st_size
            print(f"  📄 {item.name} ({size:,} bytes)")
        elif item.is_dir():
            print(f"  📁 {item.name}/")


def test_error_handling():
    """Test error handling scenarios"""
    print_section("Test 6: Error Handling")

    # Test 1: Extract with no metadata file
    print("1. Testing extraction when metadata file doesn't exist:")
    # Temporarily rename metadata file if it exists
    metadata_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    backup_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json.backup")

    metadata_existed = False
    if os.path.exists(metadata_path):
        os.rename(metadata_path, backup_path)
        metadata_existed = True

    result = extract_scraped_info("test")
    result_data = json.loads(result)

    if "error" in result_data:
        print(f"   ✓ Correctly returned error: {result_data['error']}")
    else:
        print(f"   ✗ Should have returned error")

    # Restore metadata file
    if metadata_existed:
        os.rename(backup_path, metadata_path)

    # Test 2: Extract with invalid identifier
    print("\n2. Testing extraction with invalid identifier:")
    result = extract_scraped_info("completely_invalid_id_12345")
    result_data = json.loads(result)

    if "error" in result_data:
        print(f"   ✓ Correctly returned error: {result_data['error']}")
    else:
        print(f"   ✗ Should have returned error")


def create_sample_data():
    """Create sample scraped data for testing"""
    print_section("Create Sample Test Data")

    # Create scrape directory
    os.makedirs(SCRAPE_DIR, exist_ok=True)

    # Create sample metadata
    sample_metadata = {
        "test_provider": {
            "provider_name": "test_provider",
            "url": "https://test.example.com",
            "domain": "test.example.com",
            "scraped_at": "2025-12-30T12:00:00.000000",
            "formats": ["markdown", "html"],
            "success": True,
            "content_files": {
                "markdown": os.path.join(SCRAPE_DIR, "test_provider_markdown.txt"),
                "html": os.path.join(SCRAPE_DIR, "test_provider_html.txt")
            },
            "title": "Test Provider Page",
            "description": "Sample test provider for testing purposes"
        }
    }

    # Write metadata
    metadata_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=4)

    # Create sample content files
    markdown_content = """# Test Provider

This is sample markdown content for testing.

## Features
- Feature 1
- Feature 2
- Feature 3

## Description
This is a test provider created for manual testing of the scraping system.
"""

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Provider</title>
</head>
<body>
    <h1>Test Provider</h1>
    <p>This is sample HTML content for testing.</p>
    <ul>
        <li>Feature 1</li>
        <li>Feature 2</li>
        <li>Feature 3</li>
    </ul>
</body>
</html>
"""

    with open(os.path.join(SCRAPE_DIR, "test_provider_markdown.txt"), 'w') as f:
        f.write(markdown_content)

    with open(os.path.join(SCRAPE_DIR, "test_provider_html.txt"), 'w') as f:
        f.write(html_content)

    print("✓ Sample data created successfully!")
    print(f"\nCreated files:")
    print(f"  - {metadata_path}")
    print(f"  - {os.path.join(SCRAPE_DIR, 'test_provider_markdown.txt')}")
    print(f"  - {os.path.join(SCRAPE_DIR, 'test_provider_html.txt')}")


def main():
    """Run all manual tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MCP Server Manual Test Suite" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")

    print("\nThis script tests the starter_server.py MCP server functions.")
    print("Some tests require a valid Firecrawl API key to run.\n")

    # Check if sample data exists
    metadata_path = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    if not os.path.exists(metadata_path):
        print("📝 No test data found. Creating sample data first...\n")
        create_sample_data()

    # Run tests
    test_directory_structure()
    test_metadata_structure()
    test_extract_existing_data()
    test_error_handling()
    test_scrape_single_website()
    test_scrape_multiple_websites()

    print_section("Test Suite Complete")
    print("All manual tests completed!")
    print("\nTo create fresh sample data, run:")
    print(f"  python {__file__} --create-sample\n")


if __name__ == "__main__":
    if "--create-sample" in sys.argv:
        create_sample_data()
    else:
        main()
