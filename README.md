This is practice with DSPy to optimize LLMs.

Python version: 3.12

Uses RAG and DSPy's teleprompter to optimize LLM to impersonate individuals.

修改了dsp retrieve代码
passages = [psg.get("long_text") for psg in passages.passages]
