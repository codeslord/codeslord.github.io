---
layout: post
title: Optimizing Gen-AI Applications with DSPy and Haystack - A Practical Guide
date: 2024-05-30 15:00
summary: Building Gen-AI applications often involves the challenging and time-consuming task of manually optimizing prompts. DSPy, an open-source library, addresses this by transforming prompt engineering into an optimization problem, making it more scalable and robust.
categories: General
---

<img src="https://i.ibb.co/0qyvsDg/robot-head.webp" alt="prompt engineering with dspy" border="0">


### Optimizing Gen-AI Applications with DSPy and Haystack: A Practical Guide

Building Gen-AI applications often involves the challenging task of manually optimizing prompts. DSPy, an open-source library, addresses this by turning prompt engineering into an optimization problem, making the process more scalable and robust.

#### Overview of DSPy

DSPy simplifies prompt engineering by providing abstractions like signatures and modules to define inputs and outputs for systems interacting with Large Language Models (LLMs).

##### Example: Defining a Signature

```python
class Emotion(dspy.Signature):
    """Classify emotions in a sentence."""
    sentence = dspy.InputField()
    sentiment = dspy.OutputField(desc="Possible choices: sadness, joy, love, anger, fear, surprise.")
```

This signature translates into a structured prompt for classifying emotions in a sentence.

##### Using Modules for Optimization

DSPy modules, such as `dspy.Predict` and `dspy.ChainOfThought`, define predictors with optimizable parameters. The `dspy.ChainOfThought` module, for example, asks the LLM to provide reasoning, enhancing response accuracy.

#### Optimizing Modules

To optimize a DSPy module, you need:
1. The module to be optimized.
2. A labeled training set.
3. Evaluation metrics.

The `BootstrapFewShot` optimizer searches through the training set, selecting the best examples to include in the prompt.

##### Example: Simplified BootstrapFewShot Algorithm

```python
class SimplifiedBootstrapFewShot(Teleprompter):
    def __init__(self, metric=None):
        self.metric = metric

    def compile(self, student, trainset, teacher=None):
        teacher = teacher if teacher is not None else student
        compiled_program = student.deepcopy()
        # Map predictors and bootstrap traces
        for example in trainset:
            if self.metric(example, prediction, predicted_traces):
                for predictor, inputs, outputs in predicted_traces:
                    d = dspy.Example(automated=True, **inputs, **outputs)
                    predictor_name = self.predictor2name[id(predictor)]
                    compiled_program[predictor_name].demonstrations.append(d)
        return compiled_program
```

This algorithm goes through training inputs, makes predictions, and checks if they meet the evaluation metric.

#### Building a Custom Haystack Pipeline

Using a dataset derived from PubMedQA, we can create a Haystack pipeline to retrieve and generate concise answers.

##### Example Pipeline Setup

```python
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack import Pipeline

retriever = InMemoryBM25Retriever(document_store, top_k=3)
generator = OpenAIGenerator(model="gpt-3.5-turbo")
template = """
Given the following information, answer the question.
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
```

#### Example Query and Response

```python
question = "What effects does ketamine have on rat neural stem cells?"
response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})
print(response["llm"]["replies"][0])
```

The detailed response indicates the need for more concise answers.

#### Using DSPy for Concise Answers

##### Defining the Signature and Module

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="short and precise answer")
```

```python
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def retrieve(self, question):
        results = retriever.run(query=question)
        passages = [res.content for res in results['documents']]
        return Prediction(passages=passages)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
```

##### Defining Evaluation Metrics

```python
from haystack.components.evaluators import SASEvaluator

sas_evaluator = SASEvaluator()
sas_evaluator.warm_up()

def mixed_metric(example, pred, trace=None):
    semantic_similarity = sas_evaluator.run(ground_truth_answers=[example.answer], predicted_answers=[pred.answer])["score"]
    n_words = len(pred.answer.split())
    long_answer_penalty = 0
    if 20 < n_words < 40:
        long_answer_penalty = 0.025 * (n_words - 20)
    elif n_words >= 40:
        long_answer_penalty = 0.5
    return semantic_similarity - long_answer_penalty
```

##### Compiling the Optimized Pipeline

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=mixed_metric)
compiled_rag = optimizer.compile(RAG(), trainset=trainset)
```

Re-evaluating the compiled pipeline shows improved performance, with concise answers scoring higher.

#### Final Optimized Pipeline

```python
template = static_prompt + """
---
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{question}}
Reasoning: Let's think step by step in order to
"""

new_prompt_builder = PromptBuilder(template=template)

new_retriever = InMemoryBM25Retriever(document_store, top_k=3)
new_generator = OpenAIGenerator(model="gpt-3.5-turbo")
answer_builder = AnswerBuilder(pattern="Answer: (.*)")

optimized_rag_pipeline = Pipeline()
optimized_rag_pipeline.add_component("retriever", new_retriever)
optimized_rag_pipeline.add_component("prompt_builder", new_prompt_builder)
optimized_rag_pipeline.add_component("llm", new_generator)
optimized_rag_pipeline.add_component("answer_builder", answer_builder)

optimized_rag_pipeline.connect("retriever", "prompt_builder.documents")
optimized_rag_pipeline.connect("prompt_builder", "llm")
optimized_rag_pipeline.connect("llm.replies", "answer_builder.replies")
```

Testing the optimized pipeline confirms shorter, more precise answers.

### Conclusion

By leveraging DSPy to optimize prompts in a Haystack RAG pipeline, we improved the performance by nearly 40% without manual prompt engineering. This approach allows for scalable and robust prompt optimization, enhancing the quality and efficiency of Gen-AI applications.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fcodeslord.github.io%2Fgeneral%2F2024%2F05%2F30%2FDSPy-haystack%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)