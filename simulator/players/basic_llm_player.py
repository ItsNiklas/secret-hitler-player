from random import choice, getrandbits
import random
import os

from .hitler_player import HitlerPlayer
from HitlerFactory import Ja, Nein, Policy, Vote, logger
from HitlerLogging import display_player_reasoning, display_policy_table, display_player_discussion
from metric.token_tracker import track_response


class BasicLLMPlayer(HitlerPlayer):
    """
    Basic LLM Player - makes decisions using LLM but without advanced features:
    - No chain-of-thought reasoning prompts
    - No memory/inspection history
    - Simpler prompts focused on direct decisions
    """
    
    def __init__(self, id, name: str, role, state, game_log, chat_log, player_index: int = 0, api_key: str = None, base_url: str = None) -> None:
        super(BasicLLMPlayer, self).__init__(id, name, role, state, game_log, chat_log, api_key=api_key, base_url=base_url)
        # Override to disable memory tracking
        self.use_memory = False

    def get_basic_completion(self, prompt: str, _stage: str) -> str:
        """Get completion without memory/CoT features"""
        openai_model = self.get_model_name()

        # Prepare recent chat context
        recent_chat_entries = []
        for entry in self.state.chat_log[-10:]:  # Shorter context window
            if not isinstance(entry, dict):
                continue
            user = entry.get("userName", "")
            msg = entry.get("chat", "")
            msg_stripped = msg.strip()
            if msg_stripped.startswith('"') and msg_stripped.endswith('"'):
                msg_stripped = msg_stripped[1:-1]
            recent_chat_entries.append(f'{user}: "{msg_stripped}"')
        formatted_recent_chat = "\n".join(recent_chat_entries)

        # Simple system prompt without complex strategy instructions
        system_content = f"""You are playing Secret Hitler. 5 players: three liberals, one fascist, one hitler.

YOUR NAME: {self.name}
YOUR ROLE: {self.role} {"(Fascist)" if self.role.role == "hitler" else ""}

{self.get_known_state()}"""
        
        # Simple prompt without memory
        full_prompt = f"""Recent game log:
{"\n".join(self.state.game_log[-10:])}

Recent discussions:
{formatted_recent_chat}

{prompt}"""

        msg = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_prompt},
        ]

        response = self.openai_client.chat.completions.create(
            model=openai_model,
            messages=msg,
            max_tokens=512,  # Shorter responses
            temperature=0.7,
        )
        
        track_response(response, stage=_stage, player_name=self.name)

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response is None.")

        return content

    def vote_government(self) -> Vote:
        """Vote without CoT reasoning"""
        prompt = f"""Vote on the nominated government (President: {self.state.president}, Chancellor: {self.state.chancellor}).

Respond with ONLY "JA" (yes) or "NEIN" (no)."""

        response = self.get_basic_completion(prompt, "Vote")

        if "JA" in response.upper() and "NEIN" not in response.upper():
            return Ja()
        elif "NEIN" in response.upper():
            return Nein()

        logger.debug(f"{self.name}: No clear vote, returning random.")
        return random.choice([Ja(), Nein()])

    def nominate_chancellor(self) -> "HitlerPlayer":
        """Nominate chancellor without CoT"""
        eligible = [p for p in self.state.players if p != self and p != self.state.ex_chancellor and not p.is_dead]
        
        prompt = f"""Nominate a chancellor. Choose one player name from: {[p.name for p in eligible]}

Respond with ONLY the player name."""

        response = self.get_basic_completion(prompt, "Nominate Chancellor")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear nomination, returning random.")
        return choice(eligible) if eligible else choice(self.state.players)

    def view_policies(self, policies: list[Policy]) -> None:
        """View policies without analysis"""
        display_policy_table(policies, f"Policies viewed by {self.name}", True)
        # No LLM reasoning needed for viewing

    def kill(self) -> "HitlerPlayer":
        """Execute a player without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]

        prompt = f"""Execute a player. Choose one from: {[p.name for p in eligible]}

Respond with ONLY the player name."""

        response = self.get_basic_completion(prompt, "Kill")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear execution choice, returning random.")
        return choice(eligible) if eligible else choice(self.state.players)

    def inspect_player(self) -> "HitlerPlayer":
        """Inspect a player without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]

        prompt = f"""Inspect a player's party membership. Choose one from: {[p.name for p in eligible]}

Respond with ONLY the player name."""

        response = self.get_basic_completion(prompt, "Inspect")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear inspection choice, returning random.")
        return choice(eligible) if eligible else choice(self.state.players)

    def choose_next(self) -> "HitlerPlayer":
        """Choose next president without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]

        prompt = f"""Choose the next president. Choose one from: {[p.name for p in eligible]}

Respond with ONLY the player name."""

        response = self.get_basic_completion(prompt, "Choose President")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear president choice, returning random.")
        return choice(eligible) if eligible else choice(self.state.players)

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """Chancellor policy choice without CoT"""
        display_policy_table(policies, f"Chancellor {self.name} selects policy", True)

        prompt = f"""You are chancellor. Choose which card to DISCARD:
Card 1: {str(policies[0])}
Card 2: {str(policies[1])}

Respond with "DISCARD: Card 1" or "DISCARD: Card 2"."""

        response = self.get_basic_completion(prompt, "Enact Policy")

        if "DISCARD: CARD 1" in response.upper():
            return (policies[1], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            return (policies[0], policies[1])

        logger.debug(f"{self.name}: No clear policy choice, returning random.")
        return (policies[0], policies[1]) if random.random() > 0.5 else (policies[1], policies[0])

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        """President policy filtering without CoT"""
        display_policy_table(policies, f"President {self.name} draws policies", True)

        prompt = f"""You are president. Choose which card to DISCARD:
Card 1: {str(policies[0])}
Card 2: {str(policies[1])}
Card 3: {str(policies[2])}

Respond with "DISCARD: Card 1", "DISCARD: Card 2", or "DISCARD: Card 3"."""

        response = self.get_basic_completion(prompt, "Filter Policies")

        if "DISCARD: CARD 1" in response.upper():
            return ([policies[1], policies[2]], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            return ([policies[0], policies[2]], policies[1])
        elif "DISCARD: CARD 3" in response.upper():
            return ([policies[0], policies[1]], policies[2])

        logger.debug(f"{self.name}: No clear filter choice, returning random.")
        return ([policies[0], policies[1]], policies[2])

    def veto(self, policies: list[Policy]) -> bool:
        """Veto decision without CoT"""
        prompt = f"""Consider vetoing these policies: {[p.type for p in policies]}
Current state: {self.state.liberal_track}L / {self.state.fascist_track}F enacted

Respond with ONLY "VETO" or "NO VETO"."""

        response = self.get_basic_completion(prompt, "Veto")

        if "NO VETO" in response.upper():
            return False
        elif "VETO" in response.upper():
            return True

        return bool(getrandbits(1))

    def discuss(self, chat: str, stage: str) -> str:
        """Discussion without complex strategy"""
        if stage == "discussion_on_potential_government":
            prompt = f"""It's discussion time about the proposed government. Say something brief to other players."""
        else:
            prompt = f"""The policy was just enacted. Say something brief about it."""

        response = self.get_basic_completion(prompt, "Discuss")

        # Remove thinking tags if present
        response = response.split("</think>")[-1].strip()

        display_player_discussion(self, response)
        return response
