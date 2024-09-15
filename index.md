# LLMonPy

LLMonPy is a python library that aims to make it easy to build AI systems with mixtures-of-agents response generators,
mixture-of-agent as judge and synthetic data for ICL.  The typical python program that uses LLMonPy will use teams of 
models to generate responses, then use another team to rank the responses and the best responses are used as examples
to improve the quality of the next round or response generation.  The ranking process also generates a lot of 
question/best answer/worse answer (QBaWa) data that can be used for fine-tuning models.

[LLMonPy README](https://github.com/llmonpy/llmonpy/blob/main/README.md)