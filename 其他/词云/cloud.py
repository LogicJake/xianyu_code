import jieba
import collections
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 评论数据
with open('comments.txt') as f:
    comments = f.read()

# 文本预处理  去除一些无用的字符，只提取出中文出来
comments = re.findall('[\u4e00-\u9fa5]+', comments, re.S)
comments = "".join(comments)

# 文本分词
jb_cut = jieba.cut(comments, cut_all=True)

# 读取常见的中文停用词
result_list = []
with open('stopwords.txt', encoding='utf-8') as f:
    lines = f.readlines()
    stop_words = set()
    for word in lines:
        word = word.strip()  # 去掉读取每一行数据的\n
        stop_words.add(word)

# 自定义停用词
for word in ['东西', '女尸']:
    stop_words.add(word)

# 去除停用词
for word in jb_cut:
    # 设置停用词并去除单词长度为1的词
    if word not in stop_words and len(word) > 1:
        result_list.append(word)

# 筛选后统计
word_counts = collections.Counter(result_list)
# 获取前100最高频的词
word_counts_top100 = word_counts.most_common(200)

# 绘制词云
my_cloud = WordCloud(
    background_color='white',
    width=900,
    height=600,
    max_words=200,  # 词云显示的最大词语数量
    font_path='SimHei.ttf',  # 设置字体，如果不设置可能出现中文显示为方框的情况
    max_font_size=100,  # 设置字体最大值
    min_font_size=15,  # 设置子图最小值
    random_state=2022).generate_from_frequencies(word_counts)

# 显示生成的词云图片
plt.imshow(my_cloud)
# 无坐标轴
plt.axis('off')
# 保存词云图片
plt.savefig('comment.png')
