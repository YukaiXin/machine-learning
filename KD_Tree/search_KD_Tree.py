from math import sqrt
from collections import namedtuple


result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    k = len(point) #数据维度
    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0]*k, float("inf"), 0) #python 中用float("inf")和float("-inf")表示正负无穷


        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)

        nearer_node = temp1.nearest_point
        dist = temp1.nearest_dist

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist

        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:
            nearer_node = pivot
            dist = temp_dist
            max_dist = dist
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest =temp2.nearest_point
            dist = temp2.nearest_dist

        return  result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))