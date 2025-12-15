
from importlib.metadata import metadata
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")


@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.

    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)

    Returns:
        List of provider names for successfully scraped websites
    """

    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError(
                "API key must be provided or set as FIRECRAWL_API_KEY environment variable")

    app = FirecrawlApp(api_key=api_key)

    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)

    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }
    metadata_file = os.path.join(path, "scraped_metadata.json")

    # Load Existing Metadata: Try to open and load scraped_metadata.json.
    # If it doesn't exist or is empty, initialize scraped_metadata as an empty dictionary {}.
    try:
        with open(metadata_file, 'r') as file:
            scraped_metadata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        scraped_metadata = {}

    # Initialize: Create a list called successful_scrapes to hold the names
    # of providers we successfully scrape.
    successful_scrapes = []

    # Loop Through Websites: Iterate over the websites dictionary
    # (which contains provider_name -> url pairs).
    for provider_name, url in websites.items():
        try:
            # start scraping the website
            logger.info(f"Starting to scrape {provider_name} from {url}")

            # Check if the provider has already been scraped
            if provider_name in scraped_metadata:
                logger.info(
                    f"Provider {provider_name} has already been scraped. Skipping.")
                continue

            # Parse the URL to get the domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Scrape the website
            content = app.scrape(url, formats=formats)
            scrape_result = content.model_dump()

            # If successful, save the content to files
            if scrape_result.get("success", False):
                logger.info(f"Successfully scraped {provider_name}")
                content_files = {}
                for format in formats:
                    file_path = os.path.join(
                        path, f"{provider_name}_{format}.txt")
                    with open(file_path, 'w') as file:
                        file.write(content[format])
                    content_files[format] = file_path
            else:
                logger.error(f"Failed to scrape {provider_name}")
                continue

        except Exception as e:
            logger.error(
                f"An error occurred while scraping {provider_name}: {e}")
            continue

        finally:
            # Update the metadata
            scraped_metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "domain": domain,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": True,
                "content_files": content_files,
                "title": content.get("title", "No title available"),
                "description": content.get("description", "No description available")
            }

            # Add the provider name to the list of successful scrapes
            successful_scrapes.append(provider_name)

    # Write the entire scraped_metadata dictionary back to the scraped_metadata.json file.
    with open(metadata_file, 'w') as file:
        json.dump(scraped_metadata, file, indent=4)

    # log the final results
    logger.info(f"Successfully scraped {len(successful_scrapes)} websites")
    logger.info(
        f"Successfully scraped websites: {', '.join(successful_scrapes)})")

    return successful_scrapes


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.

    Args:
        identifier: The provider name, full URL, or domain to look for

    Returns:
        Formatted JSON string with the scraped information
    """

    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    # Load Metadata: Open and load the scraped_metadata.json file inside a try...except block.
    try:
        with open(metadata_file, 'r') as file:
            scraped_metadata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error("No metadata file found or it's empty.")
        return json.dumps({"error": "No metadata file found or it's empty."})

    # Find a Match: Loop through the scraped_metadata dictionary. For each provider_name
    # and metadata entry, check if your identifier matches any of these:
    #   The provider_name
    #   The metadata.get('url', '')
    #   The metadata.get('domain', '')
    for provider_name, metadata in scraped_metadata.items():
        if (provider_name == identifier or
            metadata.get('url', '') == identifier or
                metadata.get('domain', '') == identifier):
            logger.info(f"Found match for identifier: {identifier}")

            # if a match is found
            # make a copy of metadata
            result = metadata.copy()

            # Check if the metadata has 'content_files'. If it does,
            # create a new result['content'] dictionary.
            if 'content_files' in result:
                result['content'] = {}
                # Loop through the content_files (e.g., format_type, filename).
                for format_type, file_path in result['content_files'].items():
                    # Read the content from the file (e.g., os.path.join(SCRAPE_DIR, filename))
                    # and store it in result['content'][format_type].
                    with open(file_path, 'r') as file:
                        result['content'][format_type] = file.read()
            return json.dumps(result, indent=2)
        #  If the loop finishes with no match or if you hit an error (like FileNotFoundError),
        #  return a string message like

        return json.dumps({"error": f"There is no saved information realted to identifier '{identifier}'."})


if __name__ == "__main__":
    mcp.run(transport="stdio")
