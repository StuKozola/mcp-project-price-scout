# Testing Guide for MCP Server

This document provides comprehensive information about testing the MCP server implementation.

## Test Files Overview

### 1. `test_starter_server.py`
Comprehensive automated test suite using pytest.

**Features:**
- Unit tests for individual functions
- Integration tests for complete workflows
- Mock external dependencies (Firecrawl API)
- Tests for error handling and edge cases
- Fixtures for test data and temporary directories

**Test Coverage:**
- `scrape_websites()` function
  - Successful scraping
  - Multiple providers
  - Skipping existing providers
  - API key handling (env var and parameter)
  - Error handling and exceptions
  - Custom formats
  - Metadata structure validation

- `extract_scraped_info()` function
  - Extraction by provider name
  - Extraction by URL
  - Extraction by domain
  - Error handling (missing files, invalid identifiers)
  - Partial format handling
  - Empty content handling

### 2. `test_server_manual.py`
Interactive manual testing script for hands-on testing.

**Features:**
- Manual test scenarios with detailed output
- Creates sample test data
- Validates directory structure
- Tests error handling
- No external dependencies needed (except for actual scraping)

**Test Scenarios:**
- Single website scraping (requires API key)
- Multiple website scraping
- Extracting existing data
- Metadata structure validation
- Directory structure verification
- Error handling scenarios

### 3. `pytest.ini`
Pytest configuration file.

**Configuration:**
- Test discovery patterns
- Default command-line options
- Custom test markers
- Minimum Python version

### 4. `conftest.py`
Shared pytest fixtures and configuration.

**Fixtures:**
- `project_root`: Project directory path
- `test_data_dir`: Test data directory path
- `reset_environment`: Auto-reset environment variables

## Running Tests

### Prerequisites

Install test dependencies:

```bash
# Install pytest and related packages
pip install pytest pytest-mock pytest-cov

# Or if using uv
uv pip install pytest pytest-mock pytest-cov
```

### Running Automated Tests

**Run all tests:**
```bash
pytest
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test file:**
```bash
pytest test_starter_server.py
```

**Run specific test class:**
```bash
pytest test_starter_server.py::TestScrapeWebsites
```

**Run specific test function:**
```bash
pytest test_starter_server.py::TestScrapeWebsites::test_scrape_websites_success
```

**Run tests by marker:**
```bash
pytest -m unit           # Run only unit tests
pytest -m integration    # Run only integration tests
pytest -m "not slow"     # Skip slow tests
```

**Run with coverage report:**
```bash
pytest --cov=starter_server --cov-report=html
```

### Running Manual Tests

**Run all manual tests:**
```bash
python test_server_manual.py
```

**Create sample test data:**
```bash
python test_server_manual.py --create-sample
```

## Test Structure

### Unit Tests
Test individual functions in isolation with mocked dependencies.

Example:
```python
def test_scrape_websites_success(temp_scrape_dir, mock_firecrawl_response):
    """Test successful scraping of websites"""
    websites = {"test_provider": "https://example.com/test"}
    
    with patch('starter_server.FirecrawlApp') as mock_firecrawl:
        mock_app = Mock()
        mock_app.scrape.return_value = mock_firecrawl_response
        mock_firecrawl.return_value = mock_app
        
        result = scrape_websites(websites, api_key="test_api_key")
        
        assert result == ["test_provider"]
```

### Integration Tests
Test complete workflows with multiple functions working together.

Example:
```python
def test_full_scrape_and_extract_workflow(temp_scrape_dir, mock_firecrawl_response):
    """Test the complete workflow: scrape then extract"""
    websites = {"integration_test": "https://integration.test.com/page"}
    
    # Scrape
    scraped = scrape_websites(websites, api_key="test_api_key")
    assert "integration_test" in scraped
    
    # Extract
    result_json = extract_scraped_info("integration_test")
    result = json.loads(result_json)
    assert result["provider_name"] == "integration_test"
```

## Test Fixtures

### `temp_scrape_dir`
Creates a temporary directory for scraping operations, automatically cleaned up after tests.

```python
@pytest.fixture
def temp_scrape_dir(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr('starter_server.SCRAPE_DIR', temp_dir)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### `mock_firecrawl_response`
Provides a mock Firecrawl API response for testing without external API calls.

```python
@pytest.fixture
def mock_firecrawl_response():
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "success": True,
        "markdown": "# Test Content",
        "html": "<html>Test</html>",
    }
    return mock_response
```

### `sample_metadata`
Provides sample metadata structure for testing extraction functions.

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup and teardown
- Don't rely on test execution order

### 2. Mocking External Dependencies
- Mock Firecrawl API calls
- Mock file I/O when appropriate
- Mock environment variables

### 3. Test Data Management
- Use temporary directories for test files
- Clean up after tests
- Use fixtures for reusable test data

### 4. Assertions
- Test both success and failure cases
- Verify file creation and content
- Check metadata structure
- Validate error messages

### 5. Coverage Goals
- Aim for >80% code coverage
- Test all code paths
- Include edge cases and error conditions

## Common Test Scenarios

### Testing API Key Handling
```python
def test_api_key_from_env():
    with patch.dict(os.environ, {'FIRECRAWL_API_KEY': 'test_key'}):
        result = scrape_websites({"test": "https://example.com"})
```

### Testing Error Conditions
```python
def test_missing_metadata_file():
    result = extract_scraped_info("nonexistent")
    assert "error" in json.loads(result)
```

### Testing File Operations
```python
def test_metadata_structure(temp_scrape_dir):
    # Scrape
    scrape_websites(websites, api_key="test_key")
    
    # Verify file structure
    metadata_file = os.path.join(temp_scrape_dir, "scraped_metadata.json")
    assert os.path.exists(metadata_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        assert "provider_name" in metadata
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=starter_server --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Failing Due to API Key
- Most tests use mocks and don't require real API keys
- Tests marked with `@pytest.mark.requires_api` need real credentials
- Use `pytest -m "not requires_api"` to skip these tests

### Temporary Directory Issues
- Ensure proper cleanup in fixtures
- Check permissions on test directories
- Use absolute paths in tests

### Import Errors
- Verify `conftest.py` is present
- Check that project is in Python path
- Ensure all dependencies are installed

## Coverage Reports

Generate HTML coverage report:
```bash
pytest --cov=starter_server --cov-report=html
```

Open `htmlcov/index.html` in browser to view detailed coverage.

Generate terminal coverage report:
```bash
pytest --cov=starter_server --cov-report=term-missing
```

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for individual functions
2. **Add integration tests** for workflows
3. **Update fixtures** if new test data is needed
4. **Add markers** for test categorization
5. **Update this README** with new test scenarios

Example template for new tests:
```python
class TestNewFeature:
    """Test suite for new feature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = new_feature()
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            new_feature(invalid_input)
    
    def test_edge_cases(self):
        """Test edge cases"""
        result = new_feature(edge_case_input)
        assert result == expected_output
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)
