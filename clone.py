
from typing import List

import dspy

from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dspy.retrieve.milvus_rm import MilvusRM

import requests

import pandas as pd

import json

'''
turbo = dspy.AzureOpenAI(
    api_base="https://bxaisc.openai.azure.com/",
    api_version="2023-05-15",
    model="gpt-35-turbo",
    api_key="9cd7d887a86a4f34932bd8f2231b1522"
)
'''
turbo = dspy.AzureOpenAI(
    api_base="https://bxaisc.openai.azure.com/",
    api_version="2023-05-15",
    model="gpt-4o",
    api_key="9cd7d887a86a4f34932bd8f2231b1522"
)

def embedding_function(texts: List[str]) -> List[float]:
    rsp = requests.post(url="http://124.220.49.224:9001/v1/embeddings", json={"input": texts, "model": "sensenova/piccolo-large-zh"}).json()
    embedding = [rsp.get("data")[0].get("embedding", [])]
    return embedding

doc_rm = MilvusRM(
    collection_name='sensenova__piccolo_large_zh_file_index',
    uri="https://in01-f831c960b661326.tc-ap-shanghai.vectordb.zilliz.com.cn",
    token="f0e03f078848974dd1ecc67c788b81d7c670ca2cdfe68ab9c21580a65b6e077d570829e1204a50b02c3d9e8dcffc8718b01919bc",
    embedding_function=embedding_function
)

#colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=doc_rm)


class clone(dspy.Signature):
    """ Answer questions"""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()



class Assess(dspy.Signature):
    """assess the quality of an answer to a question"""

    evaluation_question = dspy.InputField(desc="the evaluation criterion")
    answer = dspy.InputField(desc="the actual answer in dataset")
    predicted = dspy.InputField(desc="the predicted answer")

    assessment_answer = dspy.OutputField(desc="Give a score between 1 and 20 for the predicted answer on the evaluatoin criterion question. Only give number, nothing else. If unable to rate, give 1.")



class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=("请用中文回答：" + question))
        return dspy.Prediction(context=context, answer=prediction.answer)

    def to_dict(self):
        # Convert the model components to a dictionary
        return {
            "num_passages": self.retrieve.k,
            "chain_of_thought": str(self.generate_answer)
        }

    @classmethod
    def from_dict(cls, model_dict):
        # Create a new instance of the model from a dictionary
        model = cls(num_passages=model_dict["num_passages"])
        # Note: Additional steps may be required to fully restore the ChainOfThought instance
        return model



'''basic metric template (unused)'''
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM



def llm_metric(gold, pred, trace=None):
    question = gold.question
    answer = gold.answer
    predicted = pred.answer

    print(f"Test Question: {question}")
    print(f"Actual dataset Answer: {answer}")
    print(f"Predicted Answer: {predicted}")


    style = "Does the tone and style of the predicted result match the tone and style of the actual answer?"
    structure = "Does the sentence structure of the predicted result match the sentence structure of the actual answer?"
    length = "Is the length of predicted answer consistent with the length of actual answer?"


    with dspy.context(lm = turbo):
        #context = dspy.Retrieve(k=5)(question).passages
        #print(f"Retrieved context: {context}")
        style = dspy.ChainOfThought(Assess)(evaluation_question=style, answer=answer, predicted=predicted)
        structure = dspy.ChainOfThought(Assess)(evaluation_question=structure, answer=answer, predicted=predicted)
        length = dspy.ChainOfThought(Assess)(evaluation_question=length, answer=answer, predicted=predicted)


    #print(f"Faithful: {faithful.assessment_answer}")
    print(f"Style: {style.assessment_answer}")
    print(f"Structure: {structure.assessment_answer}")
    print(f"Length: {length.assessment_answer}")

    total = round((float(style.assessment_answer) + float(length.assessment_answer) + float(structure.assessment_answer))/3, 1)
    print(f"Total: {total}")

    return total


def load_rag(path):
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    if state["class_name"] == "RAG":
        instance = RAG(num_passages=state["num_passages"])
        instance.retrieve.k = state["retrieve"]["k"]
        instance.generate_answer.traces = state["generate_answer"]["traces"]
        instance.generate_answer.train = state["generate_answer"]["train"]
        instance.generate_answer.demos = state["generate_answer"]["demos"]
        return instance
    else:
        raise ValueError("Unsupported class type")



'''old code snippets'''

'''
class ImpersonateGenerate(dspy.Signature):
    """Answer questions in 高老师's style."""

    question = dspy.InputField(desc = "Question for 高老师")
    answer = dspy.OutputField(desc = "Answer question in way that sounds like 高老师")


test_example = dspy.Example(question="这个是正宗天津菜吗?")
test_pred = dspy.Example(answer="天津菜比较有代表性的是八珍豆腐、老爆三、新爆三、全爆、八大碗。这个确实没听过。")

print(f"Total: {llm_metric(test_example, test_pred)}")

evaluate = Evaluate(devset=trainset, num_threads=1, display_progress=True, display_table=5)
evaluate(RAG(), metric=llm_metric)


dspy.Predict(GenerateAnswer)(question="胡塞拖鞋军针对老美这么久了为毛一点动作都没有啊美军")

teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)


my_question = "红领巾系法？"
pred = compiled_rag(my_question)


print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")

'''