import traceback

from clone import RAG, llm_metric
import dspy
from dspy.teleprompt import BootstrapFewShot
import json
import pandas as pd
import openpyxl

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

'''remove comment to retrain RAG'''

df = pd.read_excel("QAdatasets/gen_qa.xlsx", header = None)
testdata = list(df.itertuples(index=False, name=None))

trainset = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in testdata]

uncompiled_rag = RAG()
teleprompter = BootstrapFewShot(metric=llm_metric, max_bootstrapped_demos=20)
compiled_rag = teleprompter.compile(uncompiled_rag, trainset=trainset)

# model_dict = compiled_rag.to_dict()
#
# with open('compiled_rag.json', 'w') as json_file:
#     json.dump(model_dict, json_file)
#
#
#
# with open('compiled_rag.json', 'r') as json_file:
#     model_dict = json.load(json_file)

# saved_rag = RAG.from_dict(model_dict)

user_input = ""

while user_input != "quit":
    try:
        user_input = input("Enter a question (\"quit\" to quit):")
        if user_input != "quit":
            ans = compiled_rag(user_input).answer
            print(ans)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
