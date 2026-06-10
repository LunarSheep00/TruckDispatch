import heapq
import math

_astar_cache = {}
_dijkstra_cache = {}

def astar(graph, coords, start, end):
    if (start, end) in _astar_cache:
        return _astar_cache[(start, end)]
    if start == end:
        return ([start], 0.0)

    def h(n):
        x1, y1 = coords[n]
        x2, y2 = coords[end]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    open_heap = [(h(start), 0.0, start, [start])]
    visited = {}

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        if node == end:
            result = (path, g)
            _astar_cache[(start, end)] = result
            return result
        for neighbor, weight in graph[node].items():
            ng = g + weight
            if neighbor not in visited or visited[neighbor] > ng:
                heapq.heappush(open_heap, (ng + h(neighbor), ng, neighbor, path + [neighbor]))

    return ([], float('inf'))

def dijkstra(graph, start, end):
    if (start, end) in _dijkstra_cache:
        return _dijkstra_cache[(start, end)]
    if start == end:
        return ([start], 0.0)

    heap = [(0.0, start, [start])]
    visited = {}

    while heap:
        dist, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited[node] = dist
        if node == end:
            result = (path, dist)
            _dijkstra_cache[(start, end)] = result
            return result
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(heap, (dist + weight, neighbor, path + [neighbor]))

    return ([], float('inf'))
