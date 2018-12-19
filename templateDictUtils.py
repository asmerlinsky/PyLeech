import numpy as np


def getTemplatesNorm(template_dict):
    for key, item in template_dict.items():
        print(key, np.sqrt(np.dot(item['median'], item['median'])))

def getSortedTemplateDifference(template_dict):
    cluster_list = list(template_dict.keys())
    clust_no = len(cluster_list)
    diff_list = []
    diff_clusts = []
    for i in range(clust_no):
        clust_i = cluster_list[i]
        for j in range(i+1, clust_no):
            clust_j = cluster_list[j]

            template_diff = template_dict[clust_i]['median'] - template_dict[clust_j]['median']
            template_diff = np.sqrt(np.dot(template_diff, template_diff))
            diff_list.append(template_diff)
            diff_clusts.append((clust_i, clust_j))

    sorted_diff_list, sorted_diff_clusts = (list(t) for t in zip(*sorted(zip(diff_list, diff_clusts))))
    return sorted_diff_list, sorted_diff_clusts


