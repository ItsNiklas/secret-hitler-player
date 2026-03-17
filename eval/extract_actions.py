import json
import glob
from pathlib import Path

def extract_actions(game_file):
    with open(game_file, 'r') as f:
        data = json.load(f)

    players = data.get('players', [])
    alice_idx = next((i for i, p in enumerate(players) if p['username'].startswith('Alice')), None)
    if alice_idx is None: return []

    alice_role = players[alice_idx]['role'].lower()
    
    actions = []
    
    lib_pol = 0
    fas_pol = 0
    
    for log in data.get('logs', []):
        state = {
            'role': alice_role,
            'lib_pol': lib_pol,
            'fas_pol': fas_pol,
            'pres': log.get('presidentId'),
            'chan': log.get('chancellorId')
        }
        
        # 1. Chancellor Nomination
        if log.get('presidentId') == alice_idx:
            actions.append({'type': 'nominate', 'chancellor': log.get('chancellorId'), 'state': dict(state)})
            
        # 2. Voting
        votes = log.get('votes')
        if votes and len(votes) > alice_idx:
            actions.append({'type': 'vote', 'vote': votes[alice_idx], 'state': dict(state)})
            
        # 3. Policy discard
        pres_hand = log.get('presidentHand')
        chan_hand = log.get('chancellorHand')
        enacted = log.get('enactedPolicy')
        
        if pres_hand and chan_hand and log.get('presidentId') == alice_idx:
            # what was discarded?
            # pres_hand has 3, chan_hand has 2
            discarded = list(pres_hand)
            for p in chan_hand:
                if p in discarded:
                    discarded.remove(p)
            actions.append({'type': 'pres_discard', 'hand': pres_hand, 'discarded': discarded[0] if discarded else None, 'state': dict(state)})
            
        if chan_hand and enacted and log.get('chancellorId') == alice_idx:
            discarded = list(chan_hand)
            if enacted in discarded:
                discarded.remove(enacted)
            actions.append({'type': 'chan_discard', 'hand': chan_hand, 'discarded': discarded[0] if discarded else None, 'state': dict(state)})
            
        # update state
        if enacted == 'Liberal':
            lib_pol += 1
        elif enacted == 'Fascist':
            fas_pol += 1
            
    return actions

print("Human Game Actions (Alice = KIMI25)")
for f in Path("runs-human/runsH-KIMI25").glob("*_summary.json"):
    acts = extract_actions(f)
    print(f"\nGame: {f.name}")
    for a in acts:
        print(a)
