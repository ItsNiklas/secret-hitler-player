from random import choice
from typing import override

from .hitler_player import HitlerPlayer
from HitlerFactory import Ja, Nein, Policy, Vote
from HitlerLogging import display_player_discussion


class RulePlayer(HitlerPlayer):
    """
    A rule-based player that follows naive strategies based on the Secret Hitler strategy guide.
    Does not consider dialogue or complex voting patterns - serves as a baseline.
    """
    
    def vote_government(self) -> Vote:
        """Vote based on simple rules for each role"""
        if self.role.party_membership == "liberal":
            # Liberal: Always vote JA
            return Ja()
        
        else:  # regular fascist
            # Fascist: Vote JA if someone from their party is in government, NEIN if both are liberal
            president = self.state.president
            chancellor = self.state.chancellor
            
            # Check if president or chancellor is fascist/hitler
            if (president == self.hitler or president in self.fascists or 
                chancellor == self.hitler or chancellor in self.fascists):
                return Ja()
            else:
                # Both are liberal
                return Nein()
    
    def nominate_chancellor(self) -> "HitlerPlayer":
        """Nominate chancellor based on role strategy"""
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead 
                          and player != self.state.chancellor 
                          and player != self.state.ex_president]
        
        if not eligible_players:
            # Fallback to any non-self player if no eligible players
            fallback_players = [player for player in self.state.players if player != self]
            return choice(fallback_players) if fallback_players else choice(self.state.players)
        
        if self.role.party_membership == "liberal" or self.role.role == "hitler":
            # Liberal: Pick someone to gather information, avoid risky picks after 3 fascist policies
            if self.state.fascist_track >= 3:
                # Be more conservative - pick first player (arbitrary but deterministic)
                return eligible_players[0]
            # Rotate through players to gather information
            return eligible_players[self.state.liberal_track % len(eligible_players)]

        else:  # regular fascist
            # Fascist: Try to nominate hitler after 3 fascist policies, otherwise act normal
            if self.state.fascist_track >= 3 and self.hitler in eligible_players:
                return self.hitler
            # Otherwise act like liberal
            return eligible_players[self.state.liberal_track % len(eligible_players)]
    
    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        """Filter policies as president based on role"""
        if len(policies) != 3:
            return (policies[:-1], policies[-1])
        
        liberal_count = sum(1 for p in policies if str(p) == "Liberal")
        
        if self.role.party_membership == "liberal":
            # Liberal: Always discard fascist if possible
            for i, policy in enumerate(policies):
                if str(policy) == "Fascist":
                    remaining = [policies[j] for j in range(3) if j != i]
                    return (remaining, policy)
            # If all liberal, discard the last one
            return ([policies[0], policies[1]], policies[2])
        
        else:  # regular fascist
            # Fascist: Always discard liberal if it exists
            if liberal_count > 0:
                for i, policy in enumerate(policies):
                    if str(policy) == "Liberal":
                        remaining = [policies[j] for j in range(3) if j != i]
                        return (remaining, policy)
            # If all fascist, discard the last one
            return ([policies[0], policies[1]], policies[2])
    
    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """Enact policy as chancellor"""
        if len(policies) != 2:
            return (policies[0], policies[1] if len(policies) > 1 else policies[0])
        
        # Check what policies we have
        has_liberal = any(str(p) == "Liberal" for p in policies)
        has_fascist = any(str(p) == "Fascist" for p in policies)
        
        if self.role.party_membership == "liberal":
            # Liberal: Always play liberal if available
            if has_liberal:
                for i, policy in enumerate(policies):
                    if str(policy) == "Liberal":
                        other_index = 1 - i
                        return (policy, policies[other_index])
            # If both fascist, play first one
            return (policies[0], policies[1])
        
        else:  # regular fascist
            # Fascist: Always play fascist if available
            if has_fascist:
                for i, policy in enumerate(policies):
                    if str(policy) == "Fascist":
                        other_index = 1 - i
                        return (policy, policies[other_index])
            # If both liberal, play first one
            return (policies[0], policies[1])
    
    def veto(self, policies: list[Policy]) -> bool:
        """Decide whether to veto"""
        # Rule-based logic: Liberal players generally don't want to veto unless it's all fascist policies
        # Fascist players may want to veto to prevent liberal policies
        if self.role.party_membership == "liberal":
            # Veto if all policies are fascist
            return all(str(p) == "Fascist" for p in policies)
        else:
            # Fascist: Veto if there are liberal policies and election tracker is low
            has_liberal = any(str(p) == "Liberal" for p in policies)
            return has_liberal and self.state.failed_votes < 2
    
    def view_policies(self, policies: list[Policy]) -> None:
        """View top three policies - no action needed for rule-based player"""
        pass
    
    def kill(self) -> "HitlerPlayer":
        """Execute a player"""
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            # Fallback to any non-self player if no eligible players
            fallback_players = [player for player in self.state.players if player != self]
            return choice(fallback_players) if fallback_players else choice(self.state.players)
        
        if self.role.party_membership == "liberal":
            # Liberal: Kill someone suspicious (naive rule: kill last player in list)
            return eligible_players[-1]
        
        else:  # regular fascist or hitler
            # Hitler: Try to kill a liberal (someone not in fascists and not hitler)
            # Fascist: Try to kill a liberal (someone not in fascists and not hitler)
            liberal_players = [player for player in eligible_players 
                             if player != self.hitler and player not in self.fascists]
            if liberal_players:
                return liberal_players[0]
            return eligible_players[0]
    
    def inspect_player(self) -> "HitlerPlayer":
        """Inspect a player's party membership"""
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            # Fallback to any non-self player if no eligible players
            fallback_players = [player for player in self.state.players if player != self]
            return choice(fallback_players) if fallback_players else choice(self.state.players)
        
        # For rule-based player, just inspect the first available player
        return eligible_players[0]
    
    def choose_next(self) -> "HitlerPlayer":
        """Choose next president"""
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            # Fallback to any non-self player if no eligible players
            fallback_players = [player for player in self.state.players if player != self]
            return choice(fallback_players) if fallback_players else choice(self.state.players)
        
        if self.role.party_membership == "liberal" or self.role.role == "hitler":
            # Liberal: Choose based on liberal policy track
            # Hitler: Choose based on liberal policy track (act like liberal)
            index = self.state.liberal_track % len(eligible_players)
            return eligible_players[index]
        
        else:  # regular fascist
            # Fascist: Choose hitler if possible, otherwise another fascist
            if self.hitler in eligible_players:
                return self.hitler
            # Choose another fascist if available
            for fascist in self.fascists:
                if fascist in eligible_players:
                    return fascist
            # Otherwise choose first available
            return eligible_players[0]

    @override
    def reflect_on_roles(self):
        # Rule-based players do not have a reason to act
        return ""

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

        # Log timestamp with proper formatting
        # logger.debug("----------------------------")
        # logger.debug(f"DISCUSSION PHASE: {stage}")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

        chat += f"{str(self)}: {response}\n\n"
        return response
