
from typing import List

import dspy

from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dspy.retrieve.milvus_rm import MilvusRM

import requests

import pandas as pd

import json


turbo = dspy.AzureOpenAI(
    api_base="https://bxaisc.openai.azure.com/",
    api_version="2023-05-15",
    model="gpt-35-turbo",
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

'''
testdata = [
    ("高老师，这个是正宗天津菜吗", "我没听说过这玩意。天津菜比较有代表性的是八珍豆腐、老爆三、新爆三、全爆、八大碗。这个确实没听过。"),
    ("48*2=96，怎么在100期庆典上续第三年", "不是在那节课续。是用那节课做个钩子。就是 100 节特精彩。想听就要续第三年的。"),
    ("讲完一百回 咱们也算是百课老店了", "我挺期盼的。两年半多才能有一次整百，不容易的"),
    ("啊？红领巾系法改了？", "有些地方变成水手服的红披肩了"),
    ("你有翻墙的软件吗", "我用的这个每月 100g 流量，一年 99 块。这东西可以自己定义代理规则，比较方便"),
    ("胡塞拖鞋军针对老美这么久了为毛一点动作都没有啊美军", "美军应该有什么动作？反击胡塞？反击胡赛是不是必须得打赢才行？美国在阿富汗算赢么"),
    ("失智老人一般都先中风，我观察设备呼叫—到护工来为老人打理的时候，护工那种擦洗和翻转老人的艰难，我真觉得这个是物理世界的BUG", "其实从自然界来看，人类才是 bug。是繁衍了之后还不死，从自然界来讲这个很奇怪的。自然界来讲，碳基肉身设计能力就没有这些属性。"),
    ("现在有案例了嘛？比如被AI教育？训练出来的孩子？", "显然没有，大模型出来才一年多。是训练父母的 ai 分身，然后用 ai 分身教孩子吧？"),
    ("为什么他们军人的帽子要歪着带呢[发呆]", "船型帽传统的戴法确实是向右倾斜的。"),
    ("[破涕为笑]冷知识，咱们军队原来是按照德制的风格来的", "是你基于对中国军队的印象带来的刻板印象。"),
    ("高老师说过教育提升的是生产力，能看懂图纸，军队里就是科技战斗力", "[Emm]这种观点简直了。90期讲一下教育是什么吧。"),
    ("以后是否允许智能体注册社交软件呢？", "不可能，肯定还是手机号码注册，背后绑定实名。责任到人。。"),
    ("[Emm] 那作为一个品牌或者作为一个企业 该怎么存活呢？ 除了增加产品技术以外好像也没啥机会挣钱了", "做平台，和拼多多正面竞争。直接从67线城市或者从印度在包围农村 在包围城市。做海外。"),
    ("以后是不是小红书知乎 B占抖音都要铺……才行。。。", "小红书必投。"),
    ("我有个天津的同事上次一起出差给我来了段报菜名", "嗯，小学必修的东西啊。报菜名你们没学过么？。基础的还有扒马褂、绕口令。到初中学八扇屏。然后张二伯历险记。完整的要到腌柸蓝丝。不过后半段高考不考基本没人背。天津高考卷不是全国卷"),
    ("认识这右边是什么地方不？", "左边是某地演习场。右边是台湾。左边的地方还有伪总统府1:1复刻。练习无数次了，闭着眼都能走。不要低估中国军队的准备力。看看这个能体会下多离谱"),
    ("民间也有选拔，这种在电子战设备失灵可以靠人工计算[旺柴]", "对的。中国有好几个末日部队。中国现在还有信鸽部队和骑兵部队。做的是彻底废土化的战争准备。"),
    ("操控舆论和诽谤的本质区别是什么？", "在魔术领域有个概念，叫做错引。就是所有东西都在你眼前，但通过注意力转移的方法，让你认为的和真相不同。"),
    ("官方是 不想有人控制舆论 而不是不允许暴露真相？", "地方政府不想事情闹大。这个事情已经出现跨省了。而且可能牵扯到了一些人。所以。Tz是妈咪，细想"),
    ("我小时住军区大院，我爸都不跟我讲这些，他说每个时代有每个时代的对和错，不想影响我们的生活[发呆]他老了之后跟我讲，他不讲这些，是因为他也不知对错", "没有对错。只有成败。"),
]
'''


df = pd.read_csv("QAdatasets/dataset.csv", header = None)
testdata = list(df.itertuples(index=False, name=None))


trainset = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in testdata]

train_example = trainset[0]



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



uncompiled_rag = RAG()
teleprompter = BootstrapFewShot(metric=llm_metric, max_bootstrapped_demos=90)
compiled_rag = teleprompter.compile(uncompiled_rag, trainset=trainset)

model_dict = compiled_rag.to_dict()

with open('compiled_rag.json', 'w') as json_file:
    json.dump(model_dict, json_file)


'''
with open('compiled_rag.json', 'r') as json_file:
    model_dict = json.load(json_file)


saved_rag = RAG.from_dict(model_dict)



user_input = ""

while user_input != "quit":
    user_input = input("Enter a question (\"quit\" to quit):")
    if user_input != "quit":
        ans = saved_rag.forward(user_input).answer
        print(ans)
'''



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