# LLMonPy

LLMonPy is a python library that aims to make it easy to build AI systems that generate "good enough" responses 99.9% of the time. 
It is still in the early stages of development, but you can check out src/samples
to see how it can be used.  

### how it improves responses
tools to improve quality of output
	Aggregate output from other models
	Rank output of multiple models, then use the best as examples of good answers
	Use check lists from eval
		pass/fail -- ask model to improve response for tests is fails
		comparison -- ask it to make a new response that would win a comparison
	RAG
		Have it ask for resources it needs


### install and initial test

### examples
1) simple prompt -- eval writer
2) tourney that does same thing, but multiple LLMs
3) refinement cycle