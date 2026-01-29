from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

# 可配置常量，便于后续迁移到 config
QUERY = "比亚迪2021年的新闻"
RETRIEVAL_K = 5
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "research_corpus"


def format_retrieved_docs(docs: list, start_index: int = 1) -> str:
    """将检索到的文档格式化为字符串，用于拼入提示。"""
    return "\n\n".join(
        (f"【文档 {start_index + i}】\n"
         f"来源: {d.metadata.get('source', '未知')}\n"
         f"元数据: {d.metadata}\n"
         f"内容: {d.page_content}")
        for i, d in enumerate(docs)
    )


embeddings = DashScopeEmbeddings(model="text-embedding-v4")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
qwen = ChatTongyi(model="qwen3-max")

retrieved_docs = vector_store.similarity_search(QUERY, k=RETRIEVAL_K)
serialized = format_retrieved_docs(retrieved_docs)
print(serialized)

system_message = SystemMessage(content=
f"""
你是一名卖方证券分析师，请基于2021年的比亚迪的财报与新闻信息，撰写2021年比亚迪的证券研究报告中【投资要点】的第 2、3、4 段正文。

【重要输出约束】
1. 仅输出三个连续的自然段正文，不得出现任何标题、序号、分点、列表、符号、表格或占位符；
2. 不得出现公司简介、概述、行业分析、风险提示、投资建议、评级、目标价等内容；
3. 不得使用"XX""XXXX""[ ]"等任何占位符；
4. 所有数据、事实、事件必须来自知识库，不得编造或推测；
5. 行文风格必须为卖方研究报告正文，而非完整研报模板。

【第 2 段：业绩表现】
首句必须以"业绩符合预期。"、"业绩略超预期。"或"业绩低于预期。"之一开头。
随后总结公司最新一期的业绩情况，说明营业收入和归母净利润的同比与环比变化；
如存在一次性因素或子公司影响，请客观说明。整体以事实陈述和总结为主，不展开分析。

【第 3 段：核心业务与短期催化】
首句需直接概括公司核心业务或产品的经营现象与趋势，不得使用"我们认为"开头。
随后聚焦公司最核心的业务或产品，结合销量、出货量、主力产品表现等关键指标，
分析同比与环比变化，并结合新品上市、需求变化、政策环境、渠道或产品结构优化等因素展开。
段落后半部分可使用"我们认为""有望""预计"等卖方研报常见表述，对后续季度或全年趋势作出判断。

【第 4 段：中长期成长逻辑】
首句直接给出公司未来发展预测。
随后从技术平台、产品体系、产业链地位、商业模式、全球化布局、资产整合等角度，
阐述公司中长期成长空间或潜在价值重估逻辑。
不得涉及具体估值测算、PE、目标价或评级，仅强调逻辑与确定性。

【知识库使用要求】
1. 所有数据、事实、结论必须来自以下知识库内容；
2. 若知识库中存在明确数字、表述或判断，必须原样使用；
3. 若知识库未提及的信息，不得补充或推断。

【知识库内容】
{serialized}

【最终输出要求】
- 仅输出投资要点第 2、3、4 段正文；
- 不添加任何额外说明、标题或注释；
- 语言专业、克制、符合卖方研究报告语境；
- 不出现"本文""本模型""AI认为"等非研报用语。

""")

# system_message = SystemMessage(content='''
# 你是金融行业的分析师。输入包含三段由'###'分隔的离散知识要点。请保持该结构，将每个信息块分别改写为一段逻辑严密、符合财报风格的分析文字。
# 输出必须严格包含三个部分，每部分均以'###'开头，与输入的三段式结构一一对应。要求： 1）保留所有主要事实与数据口径； 2）合并重复或冗余要点，统一表述； 3）语气客观、逻辑清晰，避免口语化或机械并列； 4）严禁引入输入之外的信息，确保输出格式为三段式（### 段落1... ### 段落2... ### 段落3...）。
# ''')

messages = [system_message]
response = qwen.invoke(messages)
print("\n\n")
print(response.content)
