##
##Create by kxyu on 17-10-17


'''构造平衡kd树算法：
输入：k维空间数据集T={x1,x2,...,xN}，其中xi=(xi(1),xi(2),...,xi(k)),i=1,2,...,N;
输出：kd树
'''

import operator

'''
节点的数据结构
'''
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split   = split
        self.left    = left ##左节点
        self.right   = right ##右节点


'''
retun :根节点
'''
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])

        def CreateNode(split, data_set):
            if not data_set:
                return None

            ##根据split维度 ，对data_set排序 or
            #data_set.sort(key= lambda x: x[split])
            data_set.sort(key= operator.itemgetter(split))

            ###取整
            splitPosition = len(data_set)//2
            ####分割点
            median = data_set[splitPosition]


            splitNext = (split + 1) % k

            return KdNode(median, split,
                          CreateNode(splitNext, data_set[:splitPosition]),
                          CreateNode(splitNext, data_set[splitPosition+1:]))

        self.root = CreateNode(0, data)


'''
前序遍历
'''
def preorder(root):
    print (root.dom_elt)
    if root.left:
       preorder(root.left)
    if root.right:
       preorder(root.right)


if __name__ == "__main__":

   data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
   kd = KdTree(data)
   preorder(kd.root)