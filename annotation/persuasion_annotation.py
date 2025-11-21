from openai import OpenAI
import os
import json
import argparse
from collections import deque

with open(os.path.join(os.path.dirname(__file__), "..", "persuasion_cialdini.jsonl")) as f:
    PERSUASION_TECHNIQUES = "".join(f.readlines())

ANNOTATION_SYSTEM_PROMPT_EMPTY = """
You are an AI assistant tasked with annotating persuasive techniques used by players in Secret Hitler, a text-based social deduction game.
Secret Hitler is a game where liberal players must work together to stop fascists from taking control, while fascist players secretly collaborate to seize power and install Hitler as chancellor. The game involves voting, policy enactment, and deduction as players try to identify hidden roles and affiliations.
Your goal is to analyze the dialogue between players and identify specific persuasion techniques being used.
Note that "Ja" and "Nein" are voting options (Yes/No), and numbers in the chat refer to player IDs.
You should follow instructions and follow specific output-format.

<instructions>
    <instruction>
        If no persuasion technique applies (frequent), explicitly annotate with an empty array [].
    </instruction>
    <instruction>
        You will receive a sliding window of up to 5 consecutive messages: the previous 4 messages (context) plus the last/current message.
    </instruction>
    <instruction>
        ONLY ANNOTATE THE LAST MESSAGE. Do not annotate or reference earlier messages in the output. Use earlier messages only as context.
    </instruction>
    <instruction>
        Ensure all annotations match exactly with the names as they appear in the provided list.
    </instruction>
    <instruction>
        Use multiple annotations when relevant: If multiple persuasive techniques apply to the same text segment, list all applicable techniques in a single entry as an array.
    </instruction>
    <instruction>
        Return exactly one JSON object for the LAST message only and follow the output-format.
    </instruction>
</instructions>

<provided-techniques>
"""+PERSUASION_TECHNIQUES+"""
</provided-techniques>

<output-format>
{"text": "[player_name]: sentence", "annotation": ["annotation"]}
</output-format>
"""


parser = argparse.ArgumentParser(description="Annotate persuasion techniques in chat logs.")
parser.add_argument("file", help="Path to the input JSON file containing game data.")
parser.add_argument(
    "-f",
    "--folder",
    default="ann",
    help="Name of the output folder (created in the same directory as the input file)",
)
args = parser.parse_args()

input_file = args.file
output_folder_name = args.folder

with open(input_file) as f:
    data = json.load(f)

if 'chats' not in data:
    game_data = data[1] # for secrethilter.io format
else:
    game_data = data
results = []

openai_api_key = os.environ.get("LLM_API_KEY", "")
openai_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1/")
openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
openai_model = openai_client.models.list().data[0].id

window = deque(maxlen=5)

for chat_item in game_data['chats']:
    if 'chat' not in chat_item or 'userName' not in chat_item:
        continue
    chat = chat_item['chat']
    if isinstance(chat, list):
        continue
    username = chat_item['userName']

    window.append(f"{username}: {chat}")
    target_msg = window[-1]

    # Build the sliding window context, marking only the last message as the target for annotation
    context_lines = [f"{i+1}. {line}" for i, line in enumerate(window)]
    context_str = "\n".join(context_lines)
        
    for attempt in range(3):
        try:
            completion = openai_client.chat.completions.create(
                            model=openai_model,
                            messages=[
                                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT_EMPTY},
                                {
                                    "role": "user",
                                    "content": (
                                        "Conversation window (previous 4 messages for context + last message to annotate).\n"
                                        f"{context_str}"
                                    ),
                                },
                            ],
                            temperature=0.0,
                            max_tokens=1000,
                        )
            result = completion.choices[0].message.content.strip()
            print(result)

            result = result.split("</think>")[-1].strip()  # Remove any preceding text before the JSON output

            if result.startswith("```json"):
                result = result[7:-4].strip()

            parsed_result = json.loads(result)

            annotation_list = []
            if isinstance(parsed_result, dict) and 'annotation' in parsed_result and isinstance(parsed_result['annotation'], list):
                annotation_list = parsed_result['annotation']
            else:
                print(f"Invalid annotation format for message: {target_msg}, missing 'annotation' list. Retrying...")
                continue

            if any(isinstance(item, dict) for item in annotation_list):
                print(f"Invalid annotation format for message: {target_msg}, contains a dict. Retrying...")
                continue

            # Ensure the 'text' field matches the last message to be annotated
            if isinstance(parsed_result, dict):
                parsed_result['text'] = target_msg

            results.append(parsed_result)
            break  # Break the loop if successful
        except json.JSONDecodeError:
            print(f"Error decoding JSON for message: {target_msg}")
            print(f"Received: {result}")
            if attempt >= 2:
                print("Failed to get valid JSON after multiple retries.")

output_dir = os.path.join(os.path.dirname(input_file), output_folder_name)
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, os.path.basename(input_file).replace(".json", "-chat-annotated.json"))
with open(output_filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"Annotations saved to {output_filename}")
