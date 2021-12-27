import csv
import math
from queue import PriorityQueue
from collections import defaultdict

edge_filename = "edges.csv"
heuristic_filename = "heuristic.csv"


def bfs(start, end):
    def addEdge(graph, u, v, w):
        graph[u].append((v, w))

    def trace(graph, start, end):
        path = [end]
        dist = 0.0
        while path[-1] != start:
            path.append(parent[path[-1]])  # add parent into list
            for v, w in graph[path[-1]]:  # add the weight of the edge
                if v == path[-2]:
                    dist += w
        path.reverse()
        return path, dist

    # generate graph
    graph = defaultdict(list)
    with open(edge_filename, newline="") as file:
        data = list(csv.reader(file))[1:]
        for row in data:
            addEdge(graph, int(row[0]), int(row[1]), float(row[2]))

    # bfs
    visited = {start}  # using set
    queue = [start]
    parent = {}
    visited_num = 0
    while len(queue) > 0:
        u = queue.pop(0)
        visited_num += 1
        if u == end:
            break
        for v, _ in graph[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
                parent[v] = u
    path, dist = trace(graph, start, end)
    return path, dist, visited_num


def dfs(start, end):
    def addEdge(graph, u, v, w):
        graph[u].append((v, w))

    def trace(graph, start, end):
        path = [end]
        dist = 0.0
        while path[-1] != start:
            path.append(parent[path[-1]])
            for v, w in graph[path[-1]]:
                if v == path[-2]:
                    dist += w
        path.reverse()
        return path, dist

    # generate graph
    graph = defaultdict(list)
    with open(edge_filename, newline="") as file:
        data = list(csv.reader(file))[1:]
        for row in data:
            addEdge(graph, int(row[0]), int(row[1]), float(row[2]))

    # dfs
    visited = {start}  # using set
    stack = [start]
    parent = {}
    visited_num = 0
    while len(stack) > 0:
        u = stack.pop(-1)
        visited_num += 1
        if u == end:
            break
        for v, _ in graph[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
                parent[v] = u
    path, dist = trace(graph, start, end)
    return path, dist, visited_num


def ucs(start, end):
    # position of w and v inversed due to PQ
    def addEdge(graph, u, v, w):
        graph[u].append((w, v))

    def trace(graph, start, end):
        path = [end]
        dist = 0.0
        parent = {}
        for key, value in weighted_parent.items():  # removes the cost stored in weighted_parent since we don't need it
            parent[key] = value[0]
        while path[-1] != start:
            path.append(parent[path[-1]])
            for w, v in graph[path[-1]]:
                if v == path[-2]:
                    dist += w
                    break
        path.reverse()
        return path, dist

    # generate graph
    graph = defaultdict(list)
    with open(edge_filename, newline="") as file:
        data = list(csv.reader(file))[1:]
        for row in data:
            addEdge(graph, int(row[0]), int(row[1]), float(row[2]))

    # ucs
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))
    weighted_parent = {}
    visited_num = 0
    while not pq.empty():
        cost, u = pq.get()
        visited_num += 1
        if u == end:
            break
        if u not in visited:
            visited.add(u)
            for w, v in graph[u]:
                if v not in visited:
                    tmp_cost = cost + w
                    pq.put((tmp_cost, v))
                    if v not in weighted_parent:
                        weighted_parent[v] = (u, tmp_cost)
                    elif weighted_parent[v][1] > tmp_cost:
                        weighted_parent[v] = (u, tmp_cost)
    path, dist = trace(graph, start, end)
    return path, dist, visited_num


def astar(start, end):
    # position of w and v inversed due to PQ
    def addEdge(graph, u, v, w):
        graph[u].append((w, v))

    def trace(graph, start, end):
        path = [end]
        dist = 0.0
        while path[-1] != start:
            path.append(parent[path[-1]])
            for w, v in graph[path[-1]]:
                if v == path[-2]:
                    dist += w
                    break
        path.reverse()
        return path, dist

    # generate graph
    graph = defaultdict(list)
    with open(edge_filename, newline="") as file:
        data = list(csv.reader(file))[1:]
        for row in data:
            addEdge(graph, int(row[0]), int(row[1]), float(row[2]))

    # generate heuristic
    heuristic = {}
    with open(heuristic_filename, newline="") as file:
        data = list(csv.reader(file))
        n = None
        for idx, node in enumerate(data[0]):
            # check which column to read
            if node.isdigit() and end == int(node):
                n = idx
                break
        if not n:  # end node ID is not in first row of heuristic.csv
            raise BaseException("End node heuristic not found")
        for row in data[1:]:
            heuristic[int(row[0])] = float(row[n])

    # a*
    visited = {start}
    pq = PriorityQueue()
    pq.put((0, start))
    parent = {}
    cost_G = {}
    cost_G[start] = 0
    visited_num = 0

    while not pq.empty():
        _, u = pq.get()
        visited_num += 1
        if u == end:
            break
        for w, v in graph[u]:
            new_cost = cost_G[u] + w
            if v not in visited or new_cost < cost_G[v]:
                visited.add(v)
                cost_G[v] = new_cost
                tmp_cost = new_cost + heuristic[v]
                pq.put((tmp_cost, v))
                parent[v] = u

    path, dist = trace(graph, start, end)
    return path, dist, visited_num


def astar_time(start, end):
    # position of w and v inversed due to PQ
    def addEdge(graph, u, v, w, speed_limit):
        graph[u].append((speed_limit, w, v))

    def trace(graph, start, end):
        path = [end]
        dist = 0.0
        kmh_to_ms = 1000 / 60 / 60
        while path[-1] != start:
            path.append(parent[path[-1]])
            for s, w, v in graph[path[-1]]:
                if v == path[-2]:
                    dist += w/(s * kmh_to_ms)
        path.reverse()
        return path, dist

    def get_heuristic(id):
        return straight_line_dist[id] / max_speed_limit

    # generate graph
    graph = defaultdict(list)
    max_speed_limit = 0.0
    with open(edge_filename, newline="") as file:
        data = list(csv.reader(file))[1:]
        for row in data:
            addEdge(graph, int(row[0]), int(row[1]),
                    float(row[2]), float(row[3]))
            max_speed_limit = max(max_speed_limit, float(row[3]))

    straight_line_dist = {}
    with open(heuristic_filename, newline="") as file:
        data = list(csv.reader(file))
        n = None
        for idx, node in enumerate(data[0]):
            if node.isdigit() and end == int(node):
                n = idx
                break
        if not n:
            raise BaseException("End node heuristic not found")
        for row in data[1:]:
            straight_line_dist[int(row[0])] = float(row[n])

    # a*
    visited = {start}
    pq = PriorityQueue()
    pq.put((0, start))
    parent = {}
    cost_G = {}
    cost_G[start] = 0
    visited_num = 0

    while not pq.empty():
        _, u = pq.get()
        visited_num += 1
        if u == end:
            break
        for s, w, v in graph[u]:
            new_cost = cost_G[u] + w/s
            if v not in visited or new_cost < cost_G[v]:
                visited.add(v)
                cost_G[v] = new_cost
                tmp_cost = new_cost + get_heuristic(v)
                pq.put((tmp_cost, v))
                parent[v] = u

    path, dist = trace(graph, start, end)
    return path, dist, visited_num


if __name__ == "__main__":
    pass
    # start = 2270143902
    # end = 1079387396
    # start = 426882161
    # end = 1737223506
    # start = 1718165260
    # end = 8513026827

    # a = astar_time(start, end)
    # a0 = astar_time_zero(start, end)
    # a1 = astar_time_SL(start, end)

    # print((a[0] == a0[0]))
    # print((a[0] == a1[0]))
    # print(a[1], a0[1], a1[1])
    # print(a[2], a0[2], a1[2])
