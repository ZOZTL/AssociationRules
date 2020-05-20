#!/usr/bin/env python3
import json

import pandas as pd
from progressbar import *

#支持度阈值
support = 0.1
#置信度阈值
confidence = 0.5
PropertyList = ['Priority', 'Location', 'Area Id', 'beat', 'Priority', 'Incident Type Id', 'Incident Type Descripe',
                'Event Number']


# 数据处理类
class OaklandCrimeStatistics:
    def __init__(self):
        # 结果文件路径
        self.resultPath = '/Users/suxiangying/Documents/Data/results'
        pass

    def dataRead(self):

        data2011 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2011.csv", encoding="utf-8")
        data2012 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2012.csv", encoding="utf-8")
        data2013 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2013.csv", encoding="utf-8")
        data2014 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2014.csv", encoding="utf-8")
        data2015 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2015.csv", encoding="utf-8")
        data2016 = pd.read_csv("/Users/suxiangying/Documents/Data/Oakland-Crime/records-for-2016.csv", encoding="utf-8")

        # 特殊数据处理
        data2012.rename(columns={"Location 1": "Location"}, inplace=True)
        data2013.rename(columns={"Location ": "Location"}, inplace=True)
        data2014.rename(columns={"Location 1": "Location"}, inplace=True)

        data2011temp = data2011[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]
        data2012temp = data2012[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]
        data2013temp = data2013[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]
        data2014temp = data2014[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]
        data2015temp = data2015[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]
        data2016temp = data2016[
            ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description",
             "Event Number"]]

        dataAll = pd.concat([data2011temp, data2012temp, data2013temp, data2014temp, data2015temp, data2016temp],
                            axis=0)

        dataAll = dataAll.dropna(how='any')

        return dataAll

    def mining(self, featureList):
        outPath = self.resultPath
        association = AssociationRules()

        dataAll = self.dataRead()
        rows = dataAll.values.tolist()

        # 将数据转为数据字典存储

        dataSet = []
        featureNames = ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id",
                        "Incident Type Description", "Event Number"]
        for dataLine in rows:
            dataSet = []
            for i, value in enumerate(dataLine):
                if not value:
                    dataSet.append((featureNames[i], 'NA'))
                else:
                    dataSet.append((featureNames[i], value))
            dataSet.append(dataSet)

        # 获取频繁项集
        freqSet, supRata = association.apriori(dataSet)
        supRataOut = sorted(supRata.items(), key=lambda d: d[1], reverse=True)
        print("supRata ", supRata)
        # 获取强关联规则列表
        strongRulesList = association.generateRules(freqSet, supRata)
        strongRulesList = sorted(strongRulesList, key=lambda x: x[3], reverse=True)
        print("strongRulesList ", strongRulesList)

        # 将频繁项集输出到结果文件
        freqSetFile = open(os.path.join(outPath, 'Frequent.json'), 'w')
        for (key, value) in supRataOut:
            resultDict = {'set': None, 'sup': None}
            setResult = list(key)
            supResult = value
            if supResult < support:
                continue
            resultDict['set'] = setResult
            resultDict['sup'] = supResult
            jsonStr = json.dumps(resultDict, ensure_ascii=False)
            freqSetFile.write(jsonStr + '\n')
        freqSetFile.close()

        # 将关联规则输出到结果文件
        rulesFile = open(os.path.join(outPath, 'rules.json'), 'w')
        for result in strongRulesList:
            resultDict = {'XSet': None, 'YSet': None, 'sup': None, 'conf': None, 'lift': None, 'jaccard': None}
            XSet, YSet, sup, conf, lift, jaccard = result
            resultDict['XSet'] = list(XSet)
            resultDict['YSet'] = list(YSet)
            resultDict['sup'] = sup
            resultDict['conf'] = conf
            resultDict['lift'] = lift
            resultDict['jaccard'] = jaccard

            jsonStr = json.dumps(resultDict, ensure_ascii=False)
            rulesFile.write(jsonStr + '\n')
        rulesFile.close()


if __name__ == "__main__":
    OaklandCrimeStatistics().mining(PropertyList)


# 关联规则算法类
class AssociationRules:
    def __init__(self):
        self.support = support
        self.confidence = confidence

    # Apriori算法
    def apriori(self, dataSet):
        # 生成单元数候选项集
        cell = self.cellGeneration(dataSet)
        dataSet = [set(data) for data in dataSet]
        F1, supRata = self.CkLowSupportFilter(dataSet, cell)
        F = [F1]
        k = 2
        while len(F[k - 2]) > 0:
            # 大于2时，合并检测
            Ck = self.aprioriGeneration(F[k - 2], k)
            Fk, support_k = self.CkLowSupportFilter(dataSet, Ck)
            supRata.update(support_k)
            F.append(Fk)
            k += 1
        return F, supRata

    # 生成单元数候选项集
    def cellGeneration(self, dataSet):
        cell = []
        progress = ProgressBar()
        for data in progress(dataSet):
            for item in data:
                if [item] not in cell:
                    cell.append([item])
        return [frozenset(item) for item in cell]

    # 过滤支持度低于阈值的项集
    def CkLowSupportFilter(self, dataset, Ck):
        CkCount = dict()
        for data in dataset:
            for cand in Ck:
                if cand.issubset(data):
                    if cand not in CkCount:
                        CkCount[cand] = 1
                    else:
                        CkCount[cand] += 1

        items = float(len(dataset))
        returnList = []
        supRata = dict()
        # 过滤非频繁项集
        for key in CkCount:
            support = CkCount[key] / items
            if support >= self.support:
                returnList.insert(0, key)
            supRata[key] = support
        return returnList, supRata

    def aprioriGeneration(self, Fk, k):  # 当候选项元素大于2时，合并时检测是否子项集满足频繁
        returnList = []
        lenFk = len(Fk)

        for i in range(lenFk):
            for j in range(i + 1, lenFk):
                # 第k-2个项相同时，将两个集合合并
                F1 = list(Fk[i])[:k - 2]
                F2 = list(Fk[j])[:k - 2]
                F1.sort()
                F2.sort()
                if F1 == F2:
                    returnList.append(Fk[i] | Fk[j])
        return returnList

    """
    产生强关联规则算法
    :param freq: 频繁项集
    :param supRata: 频繁项集对应的支持度
    :return: 强关联规则列表
    """

    def generateRules(self, freq, supRata):
        strongRulesList = []
        for i in range(1, len(freq)):
            for freqSet in freq[i]:
                H1 = [frozenset([item]) for item in freqSet]
                # 只获取有两个或更多元素的集合
                if i > 1:
                    self.rulesFromReasonedItem(freqSet, H1, supRata, strongRulesList)
                else:
                    self.calConf(freqSet, H1, supRata, strongRulesList)
        return strongRulesList

    def rulesFromReasonedItem(self, freqSet, H, supRata, strongRulesList):
        """
        H->出现在规则右部的元素列表
        """
        m = len(H[0])
        if len(freqSet) > (m + 1):
            Hmp1 = self.aprioriGeneration(H, m + 1)
            Hmp1 = self.calConf(freqSet, Hmp1, supRata, strongRulesList)
            if len(Hmp1) > 1:
                self.rulesFromReasonedItem(freqSet, Hmp1, supRata, strongRulesList)

    def calConf(self, freqSet, H, supRata, strongRulesList):  # 评估规则
        prunedH = []
        for reasonItem in H:
            sup = supRata[freqSet]
            conf = sup / supRata[freqSet - reasonItem]
            lift = conf / supRata[reasonItem]
            jaccard = sup / (supRata[freqSet - reasonItem] + supRata[reasonItem] - sup)
            if conf >= self.confidence:
                strongRulesList.append((freqSet - reasonItem, reasonItem, sup, conf, lift, jaccard))
                prunedH.append(reasonItem)
        return prunedH
