from typing import List, Tuple, Optional
from tts_audiobook_tool.dictionary_en import DictionaryEn
from tts_audiobook_tool.text_normalizer import TextNormalizer

def count_word_errors(
        normalized_source: str,
        normalized_transcript: str,
        language_code: str="",
        verbose=False
) -> int:
    """
    Text comparison text for source text and transcribed text.
    Uses a standard dynamic programming table (dp) to find the global minimum number of failures

    Costs:
        Match: 0
        Uncommon Word Match (1-to-1 or 1-to-2): 0 ("Free pass")
        Substitution: 1
        Deletion (Skip Source): 1
        Insertion (Skip Transcript): 1    
    """

    def p(s, end="\n"):
        if verbose:
            print(s, end=end)

    def sounds_the_same(a: str, b: str) -> bool:
        if language_code == "en":
            return TextNormalizer.sounds_the_same_en(a, b)
        else:
            return False

    def is_uncommon_word(word: str) -> bool:
        # for debugging only
        if len(word) == 1:
            return False 
        if language_code == "en":
            return not DictionaryEn.has(word)
        else:
            return False

    def is_match(a: str, b: str) -> bool:
        return (a == b) or sounds_the_same(a, b)

    source_words = normalized_source.split()
    transcript_words = normalized_transcript.split()
    n = len(source_words)
    m = len(transcript_words)

    # dp[i][j] stores min failures to align source[:i] and transcript[:j]
    dp: List[List[float]] = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    # parent[i][j] stores (prev_i, prev_j, action_description)
    parent: List[List[Optional[Tuple[int, int, str]]]] = [[None] * (m + 1) for _ in range(n + 1)]
    
    dp[0][0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            
            # 1. Match or Substitution (Source matches Transcript)
            if i > 0 and j > 0:
                s_word = source_words[i-1]
                t_word = transcript_words[j-1]
                
                matches = is_match(s_word, t_word)
                is_uncommon = is_uncommon_word(s_word)
                
                if matches:
                    cost = dp[i-1][j-1] + 0
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j-1, "match_direct" if s_word == t_word else "match_homophone")
                elif is_uncommon:
                    # Free pass 1-to-1
                    cost = dp[i-1][j-1] + 0
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j-1, "uncommon_pass_1")
                else:
                    # Substitution (Mismatch) -> Cost 1
                    cost = dp[i-1][j-1] + 1
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j-1, "mismatch_sub")

            # 2. Uncommon Match 1-to-2
            if i > 0 and j > 1:
                s_word = source_words[i-1]
                if is_uncommon_word(s_word):
                    cost = dp[i-1][j-2] + 0
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j-2, "uncommon_pass_2")

            # 3. Skip Source (Deletion) -> Cost 1
            if i > 0:
                cost = dp[i-1][j] + 1
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = (i-1, j, "skip_source")

            # 4. Skip Transcript (Insertion) -> Cost 1
            if j > 0:
                cost = dp[i][j-1] + 1
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = (i, j-1, "skip_transcript")

    # Reconstruct Path
    path = []
    curr_i, curr_j = n, m
    final_fails = int(dp[n][m])

    while curr_i > 0 or curr_j > 0:
        node = parent[curr_i][curr_j]
        if node is None:
            # Should not happen if reachable
            break
        prev_i, prev_j, action = node
        path.append(action)
        curr_i, curr_j = prev_i, prev_j

    path.reverse()

    current_fails = 0
    
    for action in path:
        if action == "match_direct":
            p("direct match")
        elif action == "match_homophone":
            p("homophone match")
        elif action == "uncommon_pass_1":
            p("no match but giving it a free pass because source word is uncommon")
        elif action == "uncommon_pass_2":
            p("no match (2 words) but giving it a free pass because source word is uncommon")
        elif action == "skip_source":
            current_fails += 1
            p(f"transcript out of words - fail #{current_fails}")
        elif action == "skip_transcript":
            current_fails += 1
            p(f"source out of words - fail #{current_fails}")
        elif action == "mismatch_sub":
            current_fails += 1
            p(f"mismatch - fail #{current_fails}")
            
    return final_fails

# ---

list = [
    ("a b c d", "a b", 2), # Transcript doesn't match last words in source
    ("a b", "a b c d", 2), # Transcript hallucinated extra words at the end
    ("a b c d e f g", "b d f g", 3), # Transcript missing some words
    ("a b c d", "a b x y z d", 3), # Transcript missing a word and added non-existent words:
    ("a b c", "x y z", 3) # Transcript doesn't match at all
]
for source, transcript, answer in list:
    print(source)
    print(transcript)
    print()
    num_fails = count_word_errors(source, transcript, language_code="en", verbose=False)
    print("num fails", num_fails)
    print()