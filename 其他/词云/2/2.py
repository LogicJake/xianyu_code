import wordcloud
import jieba
import matplotlib.pyplot as plt


def show(d, pic):
    wc = wordcloud.WordCloud(
        font_path='simfang.ttf',  # 加载中文字体，否则图片中文显示为方框
        background_color='white',
        mask=pic,  # 使用圆形背景
        random_state=2022)

    t = wc.generate_from_frequencies(d)

    plt.imshow(t)
    plt.axis('off')
    plt.savefig('豆瓣电影.png')


# 加载评论
with open('豆瓣评论.txt', encoding='utf-8') as f:
    txt = f.read()

# 进行分词
words = jieba.lcut(txt)

# 读取无意义的词
excludes = set()
with open('stopwords.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        excludes.add(line)

# 单词计数，跳过无意义的词
d = {}
for w in words:
    if len(w) == 1 or w in excludes:
        continue
    # 单词未在字典中，先初始化为0
    if w not in d:
        d[w] = 0
    # 计数+1
    d[w] = d[w] + 1

pic = plt.imread('circle.jpg')
show(d, pic)
