import json
from pathlib import Path

def extract_actions(game_file):
    with open(game_file, 'r') as f:
        data = json.load(f)

    players = data.get('players', [])
    # In some files _id is string, but list index is reliable if players are ordered
    # let's map id to role securely
    roles = {}
    alice_id = None
    for i, p in enumerate(players):
        pid = i
        roles[pid] = p['role'].lower()
        if p['username'].startswith('Alice'):
            alice_id = pid
            
    if alice_id is None: return []

    alice_role = roles[alice_id]
    
    actions = []
    
    lib_pol = 0
    fas_pol = 0
    tracker = 0
    
    for log in data.get('logs', []):
        pres = log.get('presidentId')
        chan = log.get('chancellorId')
        
        state_key = f"{alice_role}_L{lib_pol}_F{fas_pol}_T{tracker}"
        
        # 1. Chancellor Nomination
        if pres == alice_id:
            nominee_role = roles.get(chan, 'unknown')
            actions.append({
                'context': f"{state_key}_Nominate",
                'details': {'pres_role': alice_role},
                'action': nominee_role
            })
            
        # 2. Voting
        votes = log.get('votes')
        if votes and len(votes) > alice_id:
            pres_role = roles.get(pres, 'unknown')
            chan_role = roles.get(chan, 'unknown')
            actions.append({
                'context': f"{state_key}_Vote_P:{pres_role}_C:{chan_role}",
                'details': {},
                'action': votes[alice_id]
            })
            
            # Update tracker if vote fails
            yes_votes = sum(1 for v in votes if v)
            if yes_votes <= len(votes) / 2:
                tracker += 1
                if tracker == 3:
                    # Top deck enacted
                    enacted = log.get('enactedPolicy')
                    if enacted == 'Liberal': lib_pol += 1
                    elif enacted == 'Fascist': fas_pol += 1
                    tracker = 0
                continue
            else:
                tracker = 0
        
        # 3. Policy discard
        pres_hand = log.get('presidentHand')
        chan_hand = log.get('chancellorHand')
        enacted = log.get('enactedPolicy')
        
        if pres_hand and chan_hand and pres == alice_id:
            hand_str = "".join(sorted([p[0] for p in pres_hand]))
            discarded = list(pres_hand)
            for p in chan_hand:
                if p in discarded:
                    discarded.remove(p)
            actions.append({
                'context': f"{state_key}_PresDiscard_Hand:{hand_str}",
                'details': {},
                'action': discarded[0][0] if discarded else None
            })
            
        if chan_hand and enacted and chan == alice_id:
            hand_str = "".join(sorted([p[0] for p in chan_hand]))
            discarded = list(chan_hand)
            if enacted in discarded:
                discarded.remove(enacted)
            actions.append({
                'context': f"{state_key}_ChanDiscard_Hand:{hand_str}",
                'details': {},
                'action': discarded[0][0] if discarded else None
            })
            
        # update state
        if enacted == 'Liberal':
            lib_pol += 1
        elif enacted == 'Fascist':
            fas_pol += 1
            
    return actions

def analyze_model_consistency(human_dir, f2_dir):
    print(f"\nComparing {human_dir} against {f2_dir}")
    
    # 1. Collect human actions
    human_acts = []
    for f in Path(human_dir).glob("*_summary.json"):
        human_acts.extend(extract_actions(f))
        
    # Group by context
    human_contexts = {}
    for a in human_acts:
        ctx = a['context']
        if ctx not in human_contexts:
            human_contexts[ctx] = []
        human_contexts[ctx].append(a['action'])
        
    # 2. Collect baseline actions from self-play
    f2_acts = {}
    for f in Path(f2_dir).glob("*_summary.json"):
        for a in extract_actions(f):
            ctx = a['context']
            if ctx not in f2_acts:
                f2_acts[ctx] = []
            f2_acts[ctx].append(a['action'])
            
    # 3. Compare
    for ctx, h_actions in human_contexts.items():
        if ctx not in f2_acts:
            print(f"Context {ctx}: Human actions {h_actions}, Self-play NO DATA")
            continue
            
        f2_all = f2_acts[ctx]
        # Count frequencies
        f2_counts = {}
        for x in f2_all:
            f2_counts[x] = f2_counts.get(x, 0) + 1
        
        total = len(f2_all)
        f2_dist = {k: f"{v/total*100:.1f}% ({v})" for k, v in f2_counts.items()}
        
        print(f"Context {ctx}:")
        print(f"  Human games: {h_actions}")
        print(f"  Self-play:   {f2_dist}")

folders_to_check = [
    ("runs-human/runsH-KIMI25", "runsF2-KIMIK25"),
    ("runs-human/runsH-MISTRALSMALL", "runsF2-MISTRALSMALL")
]

for h, f2 in folders_to_check:
    analyze_model_consistency(h, f2)
analyze_model_consistency("runs-human/runsH-GPT52", "runsF2-GPT52")
