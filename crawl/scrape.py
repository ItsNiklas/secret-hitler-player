import requests
import json
import os
import re

# Clearance cookie
CLEARANCE = "B.yo5e5YCeBnsa1WHsL5ykepThPmP03gbWjRoQdyxQk-1758815679-1.2.1.1-eEP20PQSxfJBCCP8txm2YKIvHJZG6NtWH9S1sIPzeE0dszhBYtgS8fwS2ZeTZ3P.Lufvu_SWdFa2tKHPrPT__X_9xCCGVaY9MGUSvbDMtBkiZkyTZCUlSWGouo.d5Z0ZLfp2CzvgxFSGj9Lx_PhDqBcggLrhvzK6dLtRQFiNT6CG6K_uXJPeM6XnVPp1H5NcBOz1MLNV8VBH6docKKjIiNynI4qCdcvoc1WCKpEcudc"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"

# Output file for game IDs
GAME_IDS_FILE = "gameids.txt"
# Cache file for verified games
GAME_CACHE_FILE = "game_cache.json"


def detect_remake_pattern(game_id):
    """
    Detect if a game ID contains a 'Remake' + number pattern.
    
    Args:
        game_id (str): The game ID to check
        
    Returns:
        tuple: (base_name, remake_number) or (None, None) if no pattern found
    """
    # Pattern to match "Remake" followed by a number at the end
    pattern = r'^(.+)Remake(\d+)$'
    match = re.match(pattern, game_id)
    
    if match:
        base_name = match.group(1)
        remake_number = int(match.group(2))
        return base_name, remake_number
    
    return None, None


def expand_remake_versions(game_id, game_cache=None):
    """
    If a game ID is a remake, generate all previous remake versions and try one number up.
    
    Args:
        game_id (str): The game ID to expand
        game_cache (dict): Optional cache for game verification
        
    Returns:
        set: Set of all remake versions including the original
    """
    base_name, remake_number = detect_remake_pattern(game_id)
    
    if base_name is None or remake_number is None:
        # Not a remake pattern, return just the original ID
        return {game_id}
    
    remake_versions = {game_id}  # Include the original remake
    
    # Add all previous remake versions from Remake1 to Remake(n-1)
    for i in range(1, remake_number):
        previous_remake = f"{base_name}Remake{i}"
        remake_versions.add(previous_remake)
    
    # Try one number up (Remake(n+1)) - it might exist
    next_remake = f"{base_name}Remake{remake_number + 1}"
    
    # Check if the next remake exists by trying to verify it
    if game_cache is not None:
        is_genuine, game_data = verify_game_genuine(next_remake, game_cache)
        if is_genuine:
            remake_versions.add(next_remake)
            print(f"Found higher remake version: {next_remake}")
        else:
            # print(f"Higher remake version {next_remake} does not exist or is not genuine")
            pass
    else:
        # If no cache provided, just add it and let verification happen later
        remake_versions.add(next_remake)
        # print(f"Added potential higher remake version: {next_remake} (will verify later)")
    
    # print(f"Expanded remake {game_id} to {len(remake_versions)} versions: {sorted(remake_versions)}")
    
    return remake_versions


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
                print(f"Loaded {len(cache)} cached game verifications from {GAME_CACHE_FILE}")
                return cache
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading cache file: {e}")
    
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
        print(f"Saved {len(cache)} game verifications to cache")
    except IOError as e:
        print(f"Error saving cache file: {e}")


def verify_game_genuine(game_id, cache=None):
    """
    Verify if a game is genuine by checking gameSetting.avalonSH is null.
    Uses cache to avoid redundant API calls.
    
    Args:
        game_id (str): The game ID to verify
        cache (dict): Optional cache dictionary
        
    Returns:
        tuple: (is_genuine, game_data) where is_genuine is bool and game_data is dict or None
    """
    if cache is None:
        cache = {}
    
    # Check cache first
    if game_id in cache:
        game_data = cache[game_id]
        is_genuine = game_data.get("gameSetting", {}).get("avalonSH") is None
        # print(f"Cache hit for {game_id}: genuine={is_genuine}")
        return is_genuine, game_data
    
    # Query the API
    headers = {
        "User-Agent": USER_AGENT,
        "cookie": f"cf_clearance={CLEARANCE}",
    }
    
    try:
        url = f"https://secrethitler.io/gameSummary?id={game_id}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            game_data = response.json()
            
            # Cache the result
            cache[game_id] = game_data
            
            # Check if genuine (avalonSH should be null)
            is_genuine = game_data.get("gameSetting", {}).get("avalonSH") is None
            print(f"Verified {game_id}: genuine={is_genuine}")
            
            return is_genuine, game_data
        else:
            print(f"Error verifying game {game_id}: HTTP {response.status_code}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"Network error verifying game {game_id}: {e}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for game {game_id}: {e}")
        return False, None
    except Exception as e:
        print(f"Unexpected error verifying game {game_id}: {e}")
        return False, None


def scrape_profile_game_ids(profile_name, existing_ids=None, game_cache=None):
    """
    Scrape game IDs from a Secret Hitler profile using the JSON API.

    Args:
        profile_name (str): The username of the profile to scrape
        existing_ids (set): Optional set of existing IDs to avoid duplicates
        game_cache (dict): Optional cache for game verification data

    Returns:
        set: Set of verified genuine game IDs found for this profile
    """
    if existing_ids is None:
        existing_ids = set()
    if game_cache is None:
        game_cache = {}

    profile_game_ids = set()

    # Define headers with cookie authentication
    headers = {
        "User-Agent": USER_AGENT,
        "cookie": f"cf_clearance={CLEARANCE}",
    }

    try:
        # Make request to the JSON API endpoint
        url = f"https://secrethitler.io/profile?username={profile_name}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract game IDs from recentGames
            if "recentGames" in data:
                recent_games = data["recentGames"]
                total_games = len(recent_games)
                # print(f"Found {total_games} recent games for profile: {profile_name}")
                
                for game in recent_games:
                    game_id = game.get("_id")
                    if game_id:
                        # Expand remake versions if this is a remake
                        all_versions = expand_remake_versions(game_id, game_cache)
                        
                        # Verify and add all versions that aren't already processed
                        for version_id in all_versions:
                            if version_id not in existing_ids and version_id not in profile_game_ids:
                                # Verify the game is genuine
                                is_genuine, game_data = verify_game_genuine(version_id, game_cache)
                                
                                if is_genuine:
                                    profile_game_ids.add(version_id)
                                    print(f"Added verified genuine game ID: {version_id}")
                                else:
                                    # print(f"Skipped non-genuine game ID: {version_id}")
                                    pass
                            elif version_id in existing_ids or version_id in profile_game_ids:
                                # print(f"Duplicate game ID found: {version_id}")
                                pass
            else:
                print(f"No recentGames found for profile: {profile_name}")
        else:
            print(f"Error fetching profile {profile_name}: HTTP {response.status_code}")
            if response.status_code == 403:
                print("Access denied - cookie may be expired or invalid")

    except requests.exceptions.RequestException as e:
        print(f"Network error processing profile {profile_name}: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for profile {profile_name}: {e}")
    except Exception as e:
        print(f"Unexpected error processing profile {profile_name}: {e}")

    return profile_game_ids


def save_game_ids(game_ids):
    """
    Save game IDs to a file without duplicates

    Args:
        game_ids (set): Set of game IDs to save
    """
    # Load existing IDs if file exists
    existing_ids = set()
    if os.path.exists(GAME_IDS_FILE):
        with open(GAME_IDS_FILE, "r") as f:
            existing_ids = set(line.strip() for line in f if line.strip())

    # Add new IDs to existing ones
    all_ids = existing_ids.union(game_ids)

    # Write all IDs to file (no duplicates since we're using a set)
    with open(GAME_IDS_FILE, "w") as f:
        for game_id in all_ids:
            f.write(f"{game_id}\n")

    print(f"Saved {len(all_ids)} unique game IDs to {GAME_IDS_FILE}")
    print(f"Added {len(all_ids) - len(existing_ids)} new game IDs")


def main(profile_names):
    """
    Main function to scrape game IDs from multiple profiles

    Args:
        profile_names (list): List of profile names to scrape
    """
    # Load existing IDs if file exists
    existing_ids = set()
    if os.path.exists(GAME_IDS_FILE):
        with open(GAME_IDS_FILE, "r") as f:
            existing_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(existing_ids)} existing game IDs from {GAME_IDS_FILE}")

        # Check genuine status of existing IDs and update cache
    #     game_cache = load_game_cache()
    #     for game_id in list(existing_ids):
    #         all_versions = expand_remake_versions(game_id, game_cache)
    #         existing_ids.update(all_versions)
    #         for version_id in all_versions:
    #             is_genuine, game_data = verify_game_genuine(version_id, game_cache)
    #             if not is_genuine:
    #                 existing_ids.remove(version_id)
    #                 print(f"Removed non-genuine existing game ID: {version_id}")
                
    #     save_game_cache(game_cache)
    #     save_game_ids(existing_ids)
    # exit(0)

    # Load game verification cache
    game_cache = load_game_cache()

    # Scrape each profile and collect game IDs
    all_new_ids = set()
    for profile in profile_names:
        # print(f"\nScraping profile: {profile}")
        profile_ids = scrape_profile_game_ids(profile, existing_ids.union(all_new_ids), game_cache)
        all_new_ids.update(profile_ids)
        print(f"Found {len(profile_ids)} new verified game IDs for {profile}")

    # Save the updated cache
    save_game_cache(game_cache)

    # Save all IDs to file
    save_game_ids(all_new_ids.union(existing_ids))


if __name__ == "__main__":
    # List of profiles to scrape
    profiles = [
        "abcabc",
        "alphanumerical",
        "ailuro",
        "AIsUnite",
        "alwaysHigh",
        "ANTI2KHERO",
        "ambition",
        "anon49",
        "AppleBottomJeans",
        "Beyond2",
        "Bubbles546",
        "cracked",
        "crappiefish",
        "DocD",
        "Dr3",
        "DreamsRTrue",
        "Ekos",
        "Gamesolver",
        "Gamethrower",
        "gekke",
        "GioIzHawt",
        "Hitslayer",
        "ImAnAnt",
        "imbapingu",
        "ImThatGuyPal",
        "Inflated",
        "ItsSadness",
        "itszak",
        "Jailorr",
        "JaxsonGamble",
        "JeffWinger",
        "jemjey",
        "JGrasp",
        "jimboo",
        "joethebrooo",
        "johnscoutman",
        "jomancool55",
        "jscarny5",
        "jsm",
        "JusNextLib",
        "kbizzle1",
        "LiberalElite",
        "Libural",
        "MasterBaiter",
        "Mauler",
        "Maximovic96",
        "MaxOnMobile",
        "Mell0",
        "MommyOrDaddy",
        "MyMisc123",
        "nach023",
        "near1337",
        "nooothitIer",
        "NotAPrincePerry",
        "occhiolism",
        "olong",
        "omichManiac",
        "oolongboba",
        "perryy",
        "RichRobby",
        "RonniePunani",
        "rooter",
        "SekretLib123",
        "Shrauger",
        "Sk8",
        "Skate01king",
        "stafford",
        "Starkrush",
        "Tempest1K",
        "Terrain",
        "themeeman",
        "Tyrrox",
        "verygoodboy",
        "Warlock",
        "WettyFap",
        "Wowzer",
        "ZakonCrack",
        "zmean",
    ]

    main(profiles)
