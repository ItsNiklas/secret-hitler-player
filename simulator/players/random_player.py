from random import choice, getrandbits
import random

from .hitler_player import HitlerPlayer
from HitlerFactory import Ja, Nein, Policy, Vote


class RandomPlayer(HitlerPlayer):
    def vote_government(self) -> Vote:
        return random.choice([Ja(), Nein()])
    
    def nominate_chancellor(self) -> "HitlerPlayer":
        # Get eligible players (not self, not dead, not current/ex chancellor)
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead 
                          and player != self.state.chancellor 
                          and player != self.state.ex_president]
        
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        if len(policies) != 3:
            return (policies[:-1], policies[-1])
        discard_index = random.randint(0, 2)
        discarded = policies[discard_index]
        remaining = [policies[i] for i in range(3) if i != discard_index]
        return (remaining, discarded)
    
    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        if len(policies) != 2:
            return (policies[0], policies[1] if len(policies) > 1 else policies[0])
        if random.choice([True, False]):
            return (policies[0], policies[1])
        else:
            return (policies[1], policies[0])
    
    def veto(self, policies: list[Policy]) -> bool:
        return bool(getrandbits(1))
    
    def view_policies(self, policies: list[Policy]) -> None:
        pass
    
    def kill(self) -> "HitlerPlayer":
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def inspect_player(self) -> "HitlerPlayer":
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def choose_next(self) -> "HitlerPlayer":
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def discuss(self, chat: str, stage: str) -> str:
        responses = [
            "I have nothing to say.",
            "Let's see what happens.",
            "I'm not sure about this.",
            "Whatever you all think is best.",
            "I'll go with the flow.",
            "Hard to say.",
            "Could go either way.",
            "We'll see."
        ]
        return random.choice(responses)

    def reflect_on_roles(self) -> str:
        # Random player has no meaningful reflection
        return "I'm just going with my gut feeling here."
