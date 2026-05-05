Roast command: on-demand and automatic weekly прожарки of a randomly chosen chat member.

## Trigger modes

```
/roast command    — on-demand; picks a random member from chat_members
weekly job        — one random day per week (deterministic per ISO week); 12:00 UTC
auto-roast        — two consecutive offensive replies to the bot trigger an immediate roast
```

## Generation

```python
# Model: llama-3.3-70b-versatile
# Input: recent messages from the target user (unified_messages)
# Style: ≤ 2 sentences, sarcastic stand-up comedian, in Russian
# 10% chance: warm message instead of roast
```

## Auto-roast detection

```
GuardNode classifies message as MALICIOUS + explicit trigger (@mention / reply)
    → random refusal sent
    → hack attempt recorded in user_memories as "Пытался взломать бота N раз"
    → roasted_count incremented
```
