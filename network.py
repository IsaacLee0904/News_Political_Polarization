import networkx as nx

G = nx.Graph()

# 添加節點
for word in model.wv.index_to_key:
    G.add_node(word)

# 添加邊
for i, word1 in enumerate(model.wv.index_to_key):
    for j, word2 in enumerate(model.wv.index_to_key):
        if i < j:  # 避免重複添加邊
            similarity = model.wv.similarity(word1, word2)
            if similarity > threshold:  # 你可以設定一個閾值來決定何時添加邊
                G.add_edge(word1, word2, weight=similarity)

# 繪製網絡圖
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
