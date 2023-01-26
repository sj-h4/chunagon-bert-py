import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def analyze(tagged_list: list):
    # クラスタごとのカテゴリーを算出
    category_count_by_cluster = {}
    for tagged in tagged_list:
        cluster: str = f'cluster_{tagged[1]}'
        category: str = f'category_{tagged[2]}'
        if cluster not in category_count_by_cluster:
            category_count_by_cluster[cluster] = {'category_a': 0, 'category_b': 0, 'category_c': 0, 'category_d': 0, 'category_e': 0, 'category_f': 0}
        category_count_by_cluster[cluster][category] += 1
    
    # カテゴリーごとのクラスタを算出
    cluster_count_by_category = {}
    for cluster, category_count in category_count_by_cluster.items():
        for category, count in category_count.items():
            if category not in cluster_count_by_category:
                cluster_count_by_category[category] = {'cluster_0': 0, 'cluster_1': 0, 'cluster_2': 0, 'cluster_3': 0, 'cluster_4': 0, 'cluster_5': 0}
            cluster_count_by_category[category][cluster] += count

    print("Analysis with cluster")
    for cluster, category_count in category_count_by_cluster.items():
        total = sum(category_count.values())
        ratio = {k: round((v / total), 2) for k, v in category_count.items()}
        print(f'cluster: {cluster}')
        print(category_count)
        print(ratio, end='\n\n')
    print("Analysis with category")
    for category, cluster_count in cluster_count_by_category.items():
        total = sum(cluster_count.values())
        ratio = {k: round((v / total), 2) for k, v in cluster_count.items()}
        print(f'category: {category}')
        print(cluster_count)
        print(ratio, end='\n\n')
    return category_count_by_cluster, cluster_count_by_category


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


def show_graph(category_count: dict, cluster_count: dict):
    # クラスタごとのカテゴリーをグラフ化
    category_list = ['category_a', 'category_b', 'category_c', 'category_d', 'category_e', 'category_f']
    category_labels = ['A 時空間系', 'B 手段系', 'C 因果系', 'D 付帯状況系', 'E 範囲系', 'F 対比系']
    cluster_list = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5']
    cluster_labels = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5']

    category_count_list: list[dict] = [category_count[cluster] for cluster in cluster_list]
    cluster_count_list = [cluster_count[category] for category in category_list]

    plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i in range(6):
        x = 0
        if i > 2:
            x = 1
        y = i
        if i > 2:
            y = i - 3
        values = list(category_count_list[i].values())
        axs[x, y].pie(values, autopct=lambda pct: func(pct, values))
        axs[x, y].set_title(cluster_labels[i])
    fig.legend(category_labels, loc='upper right')
    fig.tight_layout()
    fig.savefig('output/cluster.png')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i in range(6):
        x = 0
        if i > 2:
            x = 1
        y = i
        if i > 2:
            y = i - 3
        values = list(cluster_count_list[i].values())
        axs[x, y].pie(values, autopct=lambda pct: func(pct, values))
        axs[x, y].set_title(category_labels[i])
    fig.legend(cluster_labels, loc='upper right')
    fig.tight_layout()
    fig.savefig('output/category.png')


if __name__ == '__main__':
    perser = argparse.ArgumentParser()
    perser.add_argument("input", help="入力テキストファイルのパス")
    args = perser.parse_args()
    input_file_path = args.input

    with open(input_file_path) as f:
        reader = csv.reader(f)
        tagged_list = [row for row in reader]
    category_count, cluster_count = analyze(tagged_list)
    show_graph(category_count, cluster_count)
