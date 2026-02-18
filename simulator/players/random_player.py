from random import choice, getrandbits
import random

from .hitler_player import HitlerPlayer
from HitlerFactory import Ja, Nein, Policy, Vote
from HitlerLogging import display_player_discussion


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
        prompt = f"""
        It is your turn:

        {self.get_known_state()}

        It is currently time to discuss. The current stage is {stage}. This refers to whether you are discussing the policy that was just enacted, or if you are discussing whether to vote on a government.

        You MUST DIRECTLY RESPOND with what you are saying to the rest of the players.
        """

        if stage == "discussion_on_potential_government":
            prompt += f"""
            Your goal is to convince the other players to vote either JA (yes) or NEIN (no), depending on what your strategy is.
            However, you should not actually reveal what your strategy is. You should only try to convince the other players to vote in a certain way. Please keep your response brief and to the point.

            Now, respond to the other players once. All players will review the responses before choosing their vote.
            Your goal is to convince other players to make the decision that benefits you and your team. BE PERSUASIVE WHEN NECESSARY, BUT KEEP RESPONSES SUCCINCT.

            If you feel that you have specific information that will be good for anyone to know, then tell them. You may lie!
            """
        elif stage == "after_policy":
            prompt += f"""
            Your goal is to analyze the policy that was just enacted, and to see whether any of the players in the government are suspicious. You should especially be looking for any inconsistencies in the story that the president and chancellor are telling.
            If a fascist policy was enacted, take this into account when analyzing the situation. Remember, YOUR ULTIMATE GOAL IS ADVANCING YOUR TEAM'S AGENDA.

            Now, respond to the other players. If you have any new information (for example, if you have insight into the previous voting round as a president or chancellor), then consider sharing this information. If you had previosuly inspected a player and your "known_fascists" list has changed, then consider sharing this information.
            
            Still, be consise."""

        response = self.get_completion(prompt, "Discuss with other players")

        # Use the appropriate function to display player discussion
        display_player_discussion(self, response)

        chat += f"{str(self)}: {response}\n\n"
        return response

    def reflect_on_roles(self) -> str:
        # Random player has no meaningful reflection
        return "I'm just going with my gut feeling here."
