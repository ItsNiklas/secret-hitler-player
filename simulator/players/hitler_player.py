import os
from typing import TYPE_CHECKING

from openai import OpenAI

from HitlerFactory import Ja, Nein, Policy, Role, Vote, logger
from HitlerLogging import *
from metric.token_tracker import track_response

if TYPE_CHECKING:
    from HitlerGameState import GameState

# from pinecone import Pinecone, ServerlessSpec

# pinecone_api_key = os.getenv('PINECONE_API_KEY')
# pc = Pinecone(api_key=pinecone_api_key)

# index_name = 'secret-hitler-strategy'
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=3072, # Replace with your model dimensions
#         metric="cosine", # Replace with your model metric
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# pinecone_index = pc.Index(index_name)
# pinecone_index = pc.Index(index_name)


class HitlerPlayer:
    # Read API key from file
    #with open("api-key.txt", "r") as file:
    #    openai_api_key = file.read().strip()

    openai_api_key = os.environ.get("LLM_API_KEY", "")
    openai_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1/")

    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # Configuration for batching behavior
    enable_parallel_processing = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"

    def __init__(
        self,
        id: int,
        name: str,
        role: Role,
        state: "GameState",
        game_log: list[str],
        chat_log: list[str],
    ) -> None:
        self.id = id
        self.name = name
        self.role = role
        self.state = state
        self.fascists: list[HitlerPlayer] = []
        self.hitler = None
        self.is_dead = False
        self.inspected_players = ""
        self.inspection = ""

    def reflect_on_roles(self) -> str:
        prompt = f"""Write a short monologue expressing your opinion about who you think might be Hitler and who you believe could be a fascist, based on your knowledge so far.

        Current game state:
        {self.get_known_state()}

        Then, clearly select ONE of the following categories that best describes the main basis for your deduction. Use the **exact format** shown: `"Reasoning Category: X"`, where X is one of the letters or a new category if needed.

        A: Recent policy (e.g., laws passed, voting outcomes)
        
        B: Probability-based reasoning (e.g., statistical likelihood, pattern recognition)
        
        C: Statements made by other players
        
        D: Random guess / intuition
        
        NONE: Doesn't fit — propose a new category"""

        response = self.get_completion(prompt, _stage="reflection")
        self.inspection += f"\n\n[Reflection after vote]\n{response}\n"
        return response

    def rapid_role_assessment(self) -> str:
        """Quick assessment of each other player's suspected role"""
        other_players = [player for player in self.state.players if player != self and not player.is_dead]

        if not other_players:
            return ""

        # Build the format with only existing players
        format_lines = []
        for player in other_players:
            format_lines.append(f"{player.name}: [Liberal/Fascist/Hitler/Unknown]")

        prompt = f"""Give an assessment of each other player's role. Be very brief and direct.

        Current game state:
        {self.get_known_state()}

        Respond in this exact format for each player:
        {"\n".join(format_lines)}
        
        Choose one option for each: Liberal, Fascist, Hitler, or Unknown.
        No explanations needed - just the quick assessment."""

        response = self.get_completion(prompt, _stage="rapid_assessment")
        self.inspection += f"\n\n[Rapid Role Assessment]\n{response}\n"
        return response

    def get_completion(self, prompt: str, _stage: str) -> str:
         # Pick the first model available
        openai_model = self.openai_client.models.list().data[0].id

        # Prepare recent chat context (all entries now structured dicts)
        recent_chat_entries = []
        for entry in self.state.chat_log[-50:]:
            if not isinstance(entry, dict):
                logger.warning(f"Chat log entry is not a dict: {entry}")
                continue
            user = entry.get("userName", "")
            msg = entry.get("chat", "")
            stage = entry.get("state")
            if stage:
                recent_chat_entries.append(f"[{stage}] {user}: {msg}")
            else:
                recent_chat_entries.append(f"{user}: {msg}")
        formatted_recent_chat = "\n".join(recent_chat_entries)

        # Merge all system elements into one system content block
        system_content = f"""You are playing the board game "Secret Hitler." You are playing a game with 5 total players.

    YOUR NAME IS: {self.name}

    This is a game about deception and strategy, your goal is to convince the other players that you are on their side, while secretly working to enact your own agenda.

    The only true source of truth is the game log and current game state you know.

    You are to FULLY EMBODY YOUR ROLE as either a Liberal, Fascist, or Hitler. Be as convincing in your character as possible, and keep your responses succinct.

    If you are a fascist, you should act as a liberal, but you still should either try to get Hitler elected (after 3 fascist policies are played) or get all the fascist policies down. Be careful, but don't be too careful (otherwise you will not win and the liberals will beat you.). You should NOT reveal who the other fascists are, though.

    As either a liberal or a fascist, you may lie. Telling the truth is usually better as a liberal, though.

    The role you have been chosen for this game is: {self.role} {"(Fascist)" if self.role.role == "hitler" else ""}.

    The previous PUBLIC game log:
    {"\n".join(self.state.game_log[-100:])}

    The previous PUBLIC discussions:
    {formatted_recent_chat}

    Your previous PRIVATE thoughts and reasoning:
    {self.inspection[-1000:]}"""

        msg = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        # import json
        # logger.warning(json.dumps(msg, indent=4))

        response = self.openai_client.chat.completions.create(
            model=openai_model,
            messages=msg,
            temperature=0.6,
            max_tokens=1000,
        )
        
        # Track token usage (without polluting game state)
        track_response(response, stage=_stage, player_name=self.name)

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response is None.")

        self.inspection += f"{content}\n"

        return content

    def get_known_state(self) -> str:
        formatted_players = ", ".join([player.name for player in self.state.players])
        formatted_fascists = (
            ", ".join([str(player) for player in self.fascists])
            if self.fascists
            else []
        )
        formatted_hitler = str(self.hitler) if self.hitler else "Unknown"

        return f"""-----------------------------------
        your name: {self.name}
        your role: {self.role}
        all players: {formatted_players}
        liberal policies enacted: {self.state.liberal_track}
        fascist policies enacted: {self.state.fascist_track}
        failed votes: {self.state.failed_votes}
        president: {self.state.president}
        ex-president: {self.state.ex_president}
        chancellor: {self.state.chancellor}
        most recent policy: {self.state.most_recent_policy}
        known fascists: {formatted_fascists}
        hitler: {formatted_hitler}
        -----------------------------------
        """

    # inspected players:{self.inspected_players}
    # veto available: {self.state.fascist_track >= 5}

    def get_knowledge(self) -> None:
        pass

    def __str__(self) -> str:
        return self.name

    @property
    def is_fascist(self) -> bool:
        return self.role.party_membership == "fascist"

    @property
    def is_hitler(self) -> bool:
        return self.role.role == "hitler"

    @property
    def knows_hitler(self) -> bool:
        return self.hitler is not None

    def __repr__(self) -> str:
        return "HitlerPlayer id:%d, name:%s, role:%s" % (self.id, self.name, self.role)

    def nominate_chancellor(self) -> "HitlerPlayer":
        """
        More random!
        :return: HitlerPlayer
        one of self.state.players
        """

        raise NotImplementedError("Player must be able to nominate a chancellor")

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        raise NotImplementedError("Player must be able to filter policies")

    def veto(self, policies: list[Policy]) -> bool:
        """
        Decide whether to veto an action or not
        :param policies: The policies currently being considered for veto
        :return: Boolean
        """
        raise NotImplementedError("Player must be able to veto an action")

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """
        Decide which of two policies to enact
        :param policies: policies
        :return: Tuple of (chosen, discarded)
        """
        raise NotImplementedError("Player must be able to enact a policy as chancellor")

    def vote_government(self) -> Vote:
        """
        Vote for the current president + chancellor combination
        :return: Vote
        """
        raise NotImplementedError("Player must be able to vote!")

    def view_policies(self, policies: list[Policy]) -> None:
        """
        What to do if you perform the presidential action to view the top three policies
        :return:
        """
        raise NotImplementedError("Player must react to view policies action")

    def kill(self) -> "HitlerPlayer":
        """
        Choose a person to kill
        :return:
        """
        raise NotImplementedError("Player must choose someone to kill")

    def inspect_player(self) -> "HitlerPlayer":
        """
        Choose a person's party membership to inspect
        :return:
        """
        raise NotImplementedError("Player must choose someone to inspect")

    def choose_next(self) -> "HitlerPlayer":
        """
        Choose the next president
        :return:
        """
        raise NotImplementedError("Player must choose next president")

    def discuss(self, chat: str, stage: str) -> str:
        """
        Start a discussion
        :return:
        """
        raise NotImplementedError("Player must discuss with other players")
