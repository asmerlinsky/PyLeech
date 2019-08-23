import numpy as np

class TemplateComparer:

    def __init__(self, template_dict):
        self.template_dict = template_dict

    def getTemplatesNorm(self):
        for key, item in self.template_dict.items():
            print(key, np.sqrt(np.dot(item['median'], item['median'])))

    def getSortedTemplateDifference(self, clust_list=None):
        if clust_list is None:
            clust_list = list(self.template_dict.keys())

        clust_no = len(clust_list)
        diff_list = []
        diff_clusts = []
        for i in range(clust_no):
            clust_i = clust_list[i]
            for j in range(i+1, clust_no):
                clust_j = clust_list[j]

                template_diff = self.template_dict[clust_i]['median'] - self.template_dict[clust_j]['median']
                template_diff = np.sqrt(np.dot(template_diff, template_diff))
                diff_list.append(template_diff)
                diff_clusts.append((clust_i, clust_j))

        self.results = [list(t) for t in zip(*sorted(zip(diff_list, diff_clusts)))]
        return self.results[0], self.results[1]

    def getSimilarTemplates(self, clust_no):
        pairs = []
        dists = []
        for i in range(len(self.results[1])):
            if clust_no in self.results[1][i]:
                 pairs.append(self.results[1][i])
                 dists.append(self.results[0][i])
        return pairs, dists