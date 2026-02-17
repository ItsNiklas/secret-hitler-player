"""Configuration loader for Secret Hitler game."""
import os
import re
import yaml
from typing import Tuple


class Config:
    """Main configuration object."""
    def __init__(self):
        # Game settings
        self.players = 5
        self.output = "output.txt"
        self.log_level = "INFO"
        self.summary_path = "./runs"
        self.simple_mode = False
        self.cutoff_rounds = 0
        self.player_types = ["LLM", "CPU", "CPU", "CPU", "CPU"]
        
        # LLM endpoints (nested dicts)
        self.llm = {
            "default": {"api_key": "", "base_url": "http://localhost:8080/v1/"},
            "llm_player": None,
            "basic_llm_player": None,
            "cpu_player": None,
            "rule_player": None,
        }
        
        # Processing
        self.enable_parallel = True
        self.max_parallel_games = 4
    
    @staticmethod
    def _expand_env_vars(value):
        """Expand environment variables in string values with support for ${VAR:-default} syntax."""
        if not isinstance(value, str):
            return value
        
        # Handle ${VAR:-default} syntax
        def replace_with_default(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.environ.get(var_name, default_value)
        
        # Replace ${VAR:-default} patterns
        value = re.sub(r'\$\{([^}:]+):-([^}]*)\}', replace_with_default, value)
        
        # Handle remaining $VAR or ${VAR} patterns
        value = os.path.expandvars(value)
        
        return value

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        config = cls()
        
        # Load game settings
        if 'game' in data:
            g = data['game']
            config.players = g.get('players', config.players)
            config.output = g.get('output', config.output)
            config.log_level = g.get('log_level', config.log_level)
            config.summary_path = g.get('summary_path', config.summary_path)
            config.simple_mode = g.get('simple_mode', config.simple_mode)
            config.cutoff_rounds = g.get('cutoff_rounds', config.cutoff_rounds)
            config.player_types = g.get('player_types', config.player_types)
        
        # Load LLM settings
        if 'llm' in data:
            llm = data['llm']
            
            if 'default' in llm:
                config.llm['default'] = {
                    'api_key': config._expand_env_vars(llm['default'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['default'].get('base_url', 'http://localhost:8080/v1/'))
                }
            
            if 'llm_player' in llm:
                config.llm['llm_player'] = {
                    'api_key': config._expand_env_vars(llm['llm_player'].get('api_key')),
                    'base_url': config._expand_env_vars(llm['llm_player'].get('base_url')),
                    'advanced_player_index': llm['llm_player'].get('advanced_player_index', 0),
                    'advanced': llm['llm_player'].get('advanced'),
                    'standard': llm['llm_player'].get('standard'),
                }
            
            if 'basic_llm_player' in llm:
                config.llm['basic_llm_player'] = {
                    'api_key': config._expand_env_vars(llm['basic_llm_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['basic_llm_player'].get('base_url', 'http://localhost:8080/v1/'))
                }
            
            if 'cpu_player' in llm:
                config.llm['cpu_player'] = {
                    'api_key': config._expand_env_vars(llm['cpu_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['cpu_player'].get('base_url', 'http://localhost:8080/v1/'))
                }
            
            if 'rule_player' in llm:
                config.llm['rule_player'] = {
                    'api_key': config._expand_env_vars(llm['rule_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['rule_player'].get('base_url', 'http://localhost:8080/v1/'))
                }
        
        # Load processing settings
        if 'processing' in data:
            p = data['processing']
            config.enable_parallel = p.get('enable_parallel', config.enable_parallel)
            config.max_parallel_games = p.get('max_parallel_games', config.max_parallel_games)
        
        return config
    
    def get_llm_endpoint(self, player_type: str, player_index: int = 0) -> Tuple[str, str]:
        """Get API key and base URL for a specific player type and index."""
        player_type = player_type.upper()
        
        if player_type == "LLM" and self.llm['llm_player']:
            lp = self.llm['llm_player']
            # Check if this is the advanced player
            if player_index == lp.get('advanced_player_index', 0) and lp.get('advanced'):
                return (lp['advanced']['api_key'], lp['advanced']['base_url'])
            # Standard LLM player
            elif lp.get('standard'):
                return (lp['standard']['api_key'], lp['standard']['base_url'])
            # Fallback to llm_player base config
            elif lp.get('api_key'):
                return (lp['api_key'], lp.get('base_url') or self.llm['default']['base_url'])
        
        elif player_type == "BASICLLM" and self.llm['basic_llm_player']:
            return (self.llm['basic_llm_player']['api_key'], self.llm['basic_llm_player']['base_url'])
        
        elif player_type == "CPU" and self.llm['cpu_player']:
            return (self.llm['cpu_player']['api_key'], self.llm['cpu_player']['base_url'])
        
        elif player_type == "RULE":
            if self.llm['rule_player']:
                return (self.llm['rule_player']['api_key'], self.llm['rule_player']['base_url'])
            # Fallback to CPU config
            elif self.llm['cpu_player']:
                return (self.llm['cpu_player']['api_key'], self.llm['cpu_player']['base_url'])
        
        # Default fallback
        return (self.llm['default']['api_key'], self.llm['default']['base_url'])

