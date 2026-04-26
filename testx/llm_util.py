import sys
sys.path.insert(0, "/d/p/tts-audiobook-tool")

from tts_audiobook_tool.llm_util import LlmUtil

"""
LlmUtil class - minimal functionality test 
"""

API_ENDPOINT_URL = "https://api.deepseek.com/v1/chat/completions"
TOKEN = "xxx"
MODEL = "deepseek-v4-flash" # "deepseek-v4-pro"
SYSTEM_PROMPT = ""
# PARAMS = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
PARAMS = {"thinking": {"type": "disabled"} }


llm = LlmUtil(
    api_endpoint_url=API_ENDPOINT_URL,
    token=TOKEN,
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    extra_params=PARAMS,
    verbose=True
)

print(f"Chatting with {MODEL}. Type 'quit' to exit, 'clear' to reset history.\n")

while True:
    try:
        user_input = input("You: ").strip()
        print()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "clear":
        llm.clear()
        print("(history cleared)\n")
        continue

    print("Assistant:")
    try:
        reply = llm.send(user_input, on_chunk=lambda chunk: print(chunk, end="", flush=True))
        print()
    except Exception as e:
        print(f"\nError: {e}")
