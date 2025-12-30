"""
Unit and integration tests for starter_server.py

This test suite covers:
- scrape_websites tool functionality
- extract_scraped_info tool functionality
- Error handling and edge cases
- File I/O operations
- Metadata management
"""

import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from typing import Dict, List

# Import the functions and module to test
import starter_server
from starter_server import scrape_websites, extract_scraped_info


@pytest.fixture
def temp_scrape_dir(monkeypatch):
    """Create a temporary directory for testing scrape operations"""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr('starter_server.SCRAPE_DIR', temp_dir)
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_firecrawl_response():
    """Mock successful Firecrawl API response"""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "success": True,
        "markdown": "# Test Content\n\nThis is test markdown content.",
        "html": "<html><body><h1>Test Content</h1><p>This is test HTML content.</p></body></html>",
    }
    mock_response.title = "Test Page Title"
    mock_response.description = "Test page description"
    return mock_response


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        "test_provider": {
            "provider_name": "test_provider",
            "url": "https://example.com/test",
            "domain": "example.com",
            "scraped_at": "2025-12-30T10:00:00.000000",
            "formats": ["markdown", "html"],
            "success": True,
            "content_files": {
                "markdown": "test_provider_markdown.txt",
                "html": "test_provider_html.txt"
            },
            "title": "Test Provider",
            "description": "Test provider description"
        },
        "another_provider": {
            "provider_name": "another_provider",
            "url": "https://another.com/page",
            "domain": "another.com",
            "scraped_at": "2025-12-30T11:00:00.000000",
            "formats": ["markdown"],
            "success": True,
            "content_files": {
                "markdown": "another_provider_markdown.txt"
            },
            "title": "Another Provider",
            "description": "Another provider description"
        }
    }


class TestScrapeWebsites:
    """Test suite for scrape_websites function"""

    def test_scrape_websites_success(self, temp_scrape_dir, mock_firecrawl_response):
        """Test successful scraping of websites"""
        websites = {
            "test_provider": "https://example.com/test"
        }

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites, api_key="test_api_key")

            assert result == ["test_provider"]
            assert "test_provider" in result
            mock_app.scrape.assert_called_once()

            # Verify metadata file was created
            metadata_file = os.path.join(
                temp_scrape_dir, "scraped_metadata.json")
            assert os.path.exists(metadata_file)

            # Verify content files were created
            markdown_file = os.path.join(
                temp_scrape_dir, "test_provider_markdown.txt")
            html_file = os.path.join(temp_scrape_dir, "test_provider_html.txt")
            assert os.path.exists(markdown_file)
            assert os.path.exists(html_file)

    def test_scrape_websites_multiple_providers(self, temp_scrape_dir, mock_firecrawl_response):
        """Test scraping multiple websites"""
        websites = {
            "provider1": "https://example1.com",
            "provider2": "https://example2.com"
        }

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites, api_key="test_api_key")

            assert len(result) == 2
            assert "provider1" in result
            assert "provider2" in result
            assert mock_app.scrape.call_count == 2

    def test_scrape_websites_skip_existing(self, temp_scrape_dir, mock_firecrawl_response):
        """Test that existing providers are skipped"""
        # Create existing metadata
        metadata = {
            "existing_provider": {
                "provider_name": "existing_provider",
                "url": "https://existing.com"
            }
        }
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        websites = {
            "existing_provider": "https://existing.com",
            "new_provider": "https://new.com"
        }

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites, api_key="test_api_key")

            # Only new_provider should be scraped
            assert len(result) == 1
            assert "new_provider" in result
            assert "existing_provider" not in result
            mock_app.scrape.assert_called_once()

    def test_scrape_websites_api_key_from_env(self, temp_scrape_dir, mock_firecrawl_response):
        """Test that API key can be loaded from environment variable"""
        websites = {"test_provider": "https://example.com"}

        with patch('starter_server.FirecrawlApp') as mock_firecrawl, \
                patch.dict(os.environ, {'FIRECRAWL_API_KEY': 'env_api_key'}):
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites)

            assert len(result) == 1
            mock_firecrawl.assert_called_once_with('env_api_key')

    def test_scrape_websites_no_api_key(self, temp_scrape_dir):
        """Test that ValueError is raised when no API key is provided"""
        websites = {"test_provider": "https://example.com"}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                scrape_websites(websites)

    def test_scrape_websites_failed_scrape(self, temp_scrape_dir):
        """Test handling of failed scrape attempts"""
        websites = {"failed_provider": "https://example.com"}

        mock_response = Mock()
        mock_response.model_dump.return_value = {"success": False}

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites, api_key="test_api_key")

            # Should return empty list if scrape failed
            assert len(result) == 0

    def test_scrape_websites_exception_handling(self, temp_scrape_dir):
        """Test handling of exceptions during scraping"""
        websites = {"error_provider": "https://example.com"}

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.side_effect = Exception("Network error")
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(websites, api_key="test_api_key")

            # Should continue and return empty list
            assert len(result) == 0

    def test_scrape_websites_custom_formats(self, temp_scrape_dir, mock_firecrawl_response):
        """Test scraping with custom formats"""
        websites = {"test_provider": "https://example.com"}

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            result = scrape_websites(
                websites, formats=['markdown'], api_key="test_api_key")

            assert len(result) == 1
            # Verify only markdown file was created
            markdown_file = os.path.join(
                temp_scrape_dir, "test_provider_markdown.txt")
            assert os.path.exists(markdown_file)

    def test_scrape_websites_metadata_structure(self, temp_scrape_dir, mock_firecrawl_response):
        """Test that metadata is saved with correct structure"""
        websites = {"test_provider": "https://example.com/test"}

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            scrape_websites(websites, api_key="test_api_key")

            metadata_file = os.path.join(
                temp_scrape_dir, "scraped_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Verify structure
            assert "test_provider" in metadata
            provider_data = metadata["test_provider"]
            assert provider_data["provider_name"] == "test_provider"
            assert provider_data["url"] == "https://example.com/test"
            assert provider_data["domain"] == "example.com"
            assert "scraped_at" in provider_data
            assert provider_data["formats"] == ['markdown', 'html']
            assert provider_data["success"] is True
            assert "content_files" in provider_data
            assert provider_data["title"] == "Test Page Title"
            assert provider_data["description"] == "Test page description"


class TestExtractScrapedInfo:
    """Test suite for extract_scraped_info function"""

    def test_extract_by_provider_name(self, temp_scrape_dir, sample_metadata):
        """Test extracting info by provider name"""
        # Setup metadata and content files
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f)

        # Create content files
        markdown_file = os.path.join(
            temp_scrape_dir, "test_provider_markdown.txt")
        html_file = os.path.join(temp_scrape_dir, "test_provider_html.txt")
        with open(markdown_file, 'w') as f:
            f.write("# Test Markdown Content")
        with open(html_file, 'w') as f:
            f.write("<html>Test HTML</html>")

        result_json = extract_scraped_info("test_provider")
        result = json.loads(result_json)

        assert "content" in result
        assert result["provider_name"] == "test_provider"
        assert result["url"] == "https://example.com/test"
        assert result["content"]["markdown"] == "# Test Markdown Content"
        assert result["content"]["html"] == "<html>Test HTML</html>"

    def test_extract_by_url(self, temp_scrape_dir, sample_metadata):
        """Test extracting info by full URL"""
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f)

        markdown_file = os.path.join(
            temp_scrape_dir, "test_provider_markdown.txt")
        html_file = os.path.join(temp_scrape_dir, "test_provider_html.txt")
        with open(markdown_file, 'w') as f:
            f.write("# Test Content")
        with open(html_file, 'w') as f:
            f.write("<html>Test</html>")

        result_json = extract_scraped_info("https://example.com/test")
        result = json.loads(result_json)

        assert result["provider_name"] == "test_provider"
        assert result["url"] == "https://example.com/test"

    def test_extract_by_domain(self, temp_scrape_dir, sample_metadata):
        """Test extracting info by domain"""
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f)

        markdown_file = os.path.join(
            temp_scrape_dir, "test_provider_markdown.txt")
        html_file = os.path.join(temp_scrape_dir, "test_provider_html.txt")
        with open(markdown_file, 'w') as f:
            f.write("# Test Content")
        with open(html_file, 'w') as f:
            f.write("<html>Test</html>")

        result_json = extract_scraped_info("example.com")
        result = json.loads(result_json)

        assert result["provider_name"] == "test_provider"
        assert result["domain"] == "example.com"

    def test_extract_no_metadata_file(self, temp_scrape_dir):
        """Test error handling when metadata file doesn't exist"""
        result_json = extract_scraped_info("nonexistent")
        result = json.loads(result_json)

        assert "error" in result
        assert "No metadata file found" in result["error"]

    def test_extract_identifier_not_found(self, temp_scrape_dir, sample_metadata):
        """Test error handling when identifier is not found"""
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f)

        result_json = extract_scraped_info("nonexistent_provider")
        result = json.loads(result_json)

        assert "error" in result
        assert "nonexistent_provider" in result["error"]
        assert "no saved information" in result["error"].lower()

    def test_extract_invalid_json_metadata(self, temp_scrape_dir):
        """Test error handling with invalid JSON in metadata file"""
        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            f.write("invalid json {{{")

        result_json = extract_scraped_info("test_provider")
        result = json.loads(result_json)

        assert "error" in result
        assert "No metadata file found" in result["error"]

    def test_extract_partial_formats(self, temp_scrape_dir):
        """Test extraction when only some formats are available"""
        metadata = {
            "partial_provider": {
                "provider_name": "partial_provider",
                "url": "https://partial.com",
                "domain": "partial.com",
                "scraped_at": "2025-12-30T10:00:00",
                "formats": ["markdown"],
                "success": True,
                "content_files": {
                    "markdown": os.path.join(temp_scrape_dir, "partial_provider_markdown.txt")
                },
                "title": "Partial Provider",
                "description": "Test description"
            }
        }

        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        markdown_file = os.path.join(
            temp_scrape_dir, "partial_provider_markdown.txt")
        with open(markdown_file, 'w') as f:
            f.write("# Markdown only")

        result_json = extract_scraped_info("partial_provider")
        result = json.loads(result_json)

        assert "content" in result
        assert "markdown" in result["content"]
        assert "html" not in result["content"]
        assert result["content"]["markdown"] == "# Markdown only"

    def test_extract_empty_content_files(self, temp_scrape_dir):
        """Test extraction when content files are empty"""
        metadata = {
            "empty_provider": {
                "provider_name": "empty_provider",
                "url": "https://empty.com",
                "domain": "empty.com",
                "content_files": {
                    "markdown": os.path.join(temp_scrape_dir, "empty_provider_markdown.txt")
                }
            }
        }

        metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        markdown_file = os.path.join(
            temp_scrape_dir, "empty_provider_markdown.txt")
        with open(markdown_file, 'w') as f:
            f.write("")

        result_json = extract_scraped_info("empty_provider")
        result = json.loads(result_json)

        assert "content" in result
        assert result["content"]["markdown"] == ""


class TestIntegration:
    """Integration tests that test the full workflow"""

    def test_full_scrape_and_extract_workflow(self, temp_scrape_dir, mock_firecrawl_response):
        """Test the complete workflow: scrape then extract"""
        websites = {
            "integration_test": "https://integration.test.com/page"
        }

        # Step 1: Scrape
        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            scraped = scrape_websites(websites, api_key="test_api_key")
            assert "integration_test" in scraped

        # Step 2: Extract by provider name
        result_json = extract_scraped_info("integration_test")
        result = json.loads(result_json)
        assert result["provider_name"] == "integration_test"
        assert "content" in result

        # Step 3: Extract by URL
        result_json = extract_scraped_info("https://integration.test.com/page")
        result = json.loads(result_json)
        assert result["provider_name"] == "integration_test"

        # Step 4: Extract by domain
        result_json = extract_scraped_info("integration.test.com")
        result = json.loads(result_json)
        assert result["provider_name"] == "integration_test"

    def test_multiple_providers_workflow(self, temp_scrape_dir, mock_firecrawl_response):
        """Test workflow with multiple providers"""
        websites = {
            "provider_a": "https://a.com",
            "provider_b": "https://b.com",
            "provider_c": "https://c.com"
        }

        with patch('starter_server.FirecrawlApp') as mock_firecrawl:
            mock_app = Mock()
            mock_app.scrape.return_value = mock_firecrawl_response
            mock_firecrawl.return_value = mock_app

            scraped = scrape_websites(websites, api_key="test_api_key")
            assert len(scraped) == 3

        # Verify all can be extracted
        for provider in ["provider_a", "provider_b", "provider_c"]:
            result_json = extract_scraped_info(provider)
            result = json.loads(result_json)
            assert result["provider_name"] == provider


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
