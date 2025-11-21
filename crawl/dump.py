from functools import cache
import requests
import json
import os
import logging
import traceback
import time
import gzip
import re
import gc
from collections import deque
from seleniumwire import webdriver
from selenium.webdriver.firefox.options import Options
import seleniumwire
import seleniumwire.webdriver
from tqdm import tqdm


logging.getLogger("seleniumwire.handler").setLevel(logging.WARNING)


class RecentLogsHandler(logging.Handler):
    """Custom logging handler that keeps only the latest N log messages"""
    
    def __init__(self, max_logs=20):
        super().__init__()
        self.max_logs = max_logs
        self.recent_logs = deque(maxlen=max_logs)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.recent_logs.append(msg)
            # Use tqdm.write to avoid interfering with progress bar
            tqdm.write(msg)
        except Exception:
            pass
    
    def get_recent_logs(self):
        return list(self.recent_logs)


# Set up logging with custom handler
logger = logging.getLogger("sh_crawler")
logger.setLevel(logging.INFO)

# Create custom handler for recent logs
recent_handler = RecentLogsHandler(max_logs=20)
recent_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(recent_handler)

# Disable default handlers to avoid duplicate output
logger.propagate = False

# URL and cookie constants
SOCKETIO_BASE_URL = "https://secrethitler.io"
USERAGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"
CLEARANCE = ".p1aFpwhADuFZy8CJufj_sR82KVus3gsyA7sOdC.QME-1758887810-1.2.1.1-9pMQObO3KUtAP5U6grE0u0gtMBryBOJuK6raGdsxYbk3U3ggPAig1CvP2sQkUeAhhOGzpP9oDrG2ebvOBZx3x1ehhib6M1OpUMH2wbtq8hCFRAKlQqm5OGx0Hl2xLxyPjEX2X2t9NKCbDGGTCv7cOZ7UiNCt1wuhFCNqc9SPN1.Bt9JV.6ezk.o_.geBvGUWZ4UzSxRfl4E9eNzcibHtBdbxmPeIp7cYHpwbmNEL1xkHQcFp8PTaNCxJf_CnFs0I"

# Cache file for game summaries (same as in scrape.py)
GAME_CACHE_FILE = "game_cache.json"


def load_game_cache():
    """
    Load the game verification cache from disk.
    
    Returns:
        dict: Cache dictionary with game_id -> game_data mapping
    """
    if os.path.exists(GAME_CACHE_FILE):
        try:
            with open(GAME_CACHE_FILE, "r") as f:
                cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached game summaries from {GAME_CACHE_FILE}")
                return cache
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading cache file: {e}")
    
    return {}


def save_game_cache(cache):
    """
    Save the game verification cache to disk.
    
    Args:
        cache (dict): Cache dictionary to save
    """
    try:
        with open(GAME_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved {len(cache)} game summaries to cache")
    except IOError as e:
        logger.error(f"Error saving cache file: {e}")


def fetch_game_summary(game_id, output_dir="summaries", game_cache=None):
    """Fetch the game summary via REST API or from cache"""
    if game_cache is None:
        game_cache = {}
    
    # Check if file already exists
    output_file = os.path.join(output_dir, f"{game_id}_summary.json")
    if os.path.exists(output_file):
        logger.info(f"Game summary for {game_id} already exists. Skipping...")
        return None

    # Check cache first
    if game_id in game_cache:
        game_data = game_cache[game_id]
        logger.info(f"Cache hit for {game_id}: using cached data")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save cached data to file
            with open(output_file, "w") as f:
                json.dump(game_data, f, indent=4)
            
            logger.info(f"Successfully saved cached game summary for {game_id} to {output_file}")
            return game_data
            
        except Exception as e:
            logger.error(f"Error saving cached data for {game_id}: {e}")
            return None    # API endpoint for the game summary
    summary_url = f"https://secrethitler.io/gameSummary?id={game_id}"

    headers = {
        "User-Agent": USERAGENT,
        "Cookie": f"cf_clearance={CLEARANCE};",
    }

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Make the request
        logger.info(f"Fetching game summary for {game_id} from API...")
        response = requests.get(summary_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        game_data = response.json()

        # Cache the result
        game_cache[game_id] = game_data

        # Save the data to a file in the specified directory
        with open(output_file, "w") as f:
            json.dump(game_data, f, indent=4)

        logger.info(
            f"Successfully fetched and saved game summary for {game_id} to {output_file}"
        )
        return game_data

    except Exception as e:
        logger.error(f"Error fetching game summary for {game_id}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None


def capture_xhr_traffic(game_id, driver, output_dir="replay_data"):
    """
    Captures all XHR network traffic from the game replay page using Selenium with Firefox

    Args:
        game_id: The ID of the game to capture data for
        output_dir: Directory to save the captured data

    Returns:
        Dictionary of captured XHR responses or None if failed
    """
    replay_url = f"https://secrethitler.io/observe/#/replay/{game_id}"

    # Check if file already exists
    output_file = os.path.join(output_dir, f"{game_id}_xhr_data.json")
    
    data = None  # Initialize data variable

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Aggressively clear previous requests and force garbage collection
        driver.backend.storage.clear_requests()
        gc.collect()

        # Now navigate to the replay URL
        logger.info(f"Loading replay page for game {game_id}...")
        driver.get(replay_url)

        # Wait for content to load and interact with the replay
        logger.info(f"Waiting for game replay data to load for {game_id}...")
        time.sleep(4)  # Increase wait time to ensure all data is loaded

        success = False

        for request in driver.requests:
            if request.response and "socket" in request.url:
                # Try to decompress the response body if it's gzip-compressed
                response_body = request.response.body

                if response_body:
                    try:
                        # Check for gzip compression
                        content_encoding = request.response.headers.get(
                            "Content-Encoding", ""
                        ).lower()
                        if "gzip" in content_encoding:
                            data = gzip.decompress(response_body).decode(
                                "utf-8", errors="ignore"
                            )

                            data = re.sub(r"(\d+:)?\d+:\d+(?=\[)", "", data)

                            if '"replayGameData"' in data:
                                success = True
                                with open(output_file, "w") as f:
                                    json.dump(json.loads(data), f, indent=4)
                                logger.info(
                                    f"Successfully saved replay data to {output_file}"
                                )
                                break

                    except Exception as e:
                        logger.warning(f"Error decompressing response body: {e}")

        # Clear driver request storage again
        driver.backend.storage.clear_requests()
        gc.collect()  # Force garbage collection

        if not success:
            logger.warning(
                f"Did not find any XHR data for game {game_id}. Please check the game ID or the replay URL."
            )
            return []

        # Return minimal data to indicate success
        return ["success"]

    except Exception as e:
        logger.error(f"Error capturing XHR data for {game_id}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    gameids = []

    with open("gameids.txt", "r") as f:
        for line in f:
            gameids.append(line.strip())

    output_directory = "summaries"
    xhr_output_directory = "replay_data"

    # Load game cache
    # game_cache = load_game_cache()

    # Fetch game summaries (uncomment to enable)
    # with tqdm(total=len(gameids), desc="Fetching summaries", unit="game") as pbar:
    #     for i, game_id in enumerate(gameids):
    #         pbar.set_description(f"Fetching summary {game_id}")
    #         # First fetch the game summary via REST API or cache
    #         logger.info(f"Fetching summary for game {game_id}...")
    #         game_data = fetch_game_summary(game_id, output_directory, game_cache)
    #         if game_data:
    #             logger.info(f"Game summary retrieved with {len(game_data)} fields of data")
    #             del game_data  # Clear from memory immediately
            
    #         pbar.update(1)
            
    #         # Force garbage collection every 10 items
    #         if (i + 1) % 10 == 0:
    #             gc.collect()
    
    # Save updated cache after processing summaries
    # save_game_cache(game_cache)

    # Then capture XHR traffic for the game replay
    # Set up Firefox options for headless browsing
    # Define headers
    headers = {
        "User-Agent": USERAGENT,
        "Cookie": f"cf_clearance={CLEARANCE};",
    }

    # Set up Firefox options with aggressive memory management
    options = Options()
    options.add_argument(f"user-agent={headers['User-Agent']}")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.set_preference("browser.cache.disk.enable", False)
    options.set_preference("browser.cache.memory.enable", False)
    options.set_preference("browser.cache.offline.enable", False)
    options.set_preference("network.http.use-cache", False)
    options.set_preference("browser.sessionhistory.max_total_viewers", 0)
    options.set_preference("browser.sessionstore.max_tabs_undo", 0)
    options.set_preference("browser.sessionstore.max_windows_undo", 0)
    options.set_preference("browser.tabs.animate", False)
    options.set_preference("browser.tabs.warnOnClose", False)
    options.set_preference("network.http.speculative-parallel-limit", 0)
    options.set_preference("network.dns.disablePrefetch", True)
    options.set_preference("network.prefetch-next", False)

    # Initialize the Firefox WebDriver with options

    # Use tqdm for progress tracking
    with tqdm(total=len(gameids), desc="Processing games", unit="game") as pbar:
        for i, game_id in enumerate(gameids):
            pbar.set_description(f"Processing {game_id}")
            
            # if gameid already exists, skip
            if os.path.exists(os.path.join(xhr_output_directory, f"{game_id}_xhr_data.json")):
                # logger.info(f"XHR data for game {game_id} already exists. Skipping...")
                pbar.update(1)
                continue
            
            driver = None
            try:
                driver = webdriver.Firefox(options=options)

                # Add cloudflare clearance cookie
                driver.get("https://secrethitler.io")
                driver.add_cookie(
                    {"name": "cf_clearance", "value": CLEARANCE, "domain": ".secrethitler.io"}
                )

                logger.info(f"Capturing XHR traffic for game {game_id}...")
                xhr_data = capture_xhr_traffic(game_id, driver, xhr_output_directory)
                
                if xhr_data:
                    logger.info(f"XHR data for {game_id} captured with {len(xhr_data)} requests")
                
                # Clear xhr_data from memory immediately
                del xhr_data
                
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
            
            finally:
                # Ensure driver is always closed
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                    del driver
                
                # Force garbage collection every 5 games to prevent memory buildup
                if (i + 1) % 5 == 0:
                    gc.collect()
                    logger.info(f"Forced garbage collection after {i + 1} games")
            
            pbar.update(1)
    
    # Display final summary
    tqdm.write(f"\n=== Processing Complete ===")
    tqdm.write(f"Total games processed: {len(gameids)}")
    tqdm.write(f"Recent logs:")
    for log_msg in recent_handler.get_recent_logs()[-10:]:  # Show last 10 logs
        tqdm.write(f"  {log_msg}")
