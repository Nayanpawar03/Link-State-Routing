import base64
import io
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, make_response
import networkx as nx
import heapq
import matplotlib
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

matplotlib.use('Agg')

app = Flask(__name__)

# Global Graph
graph = nx.Graph()

# Function to Add Router (Node)
@app.route('/add_router', methods=['POST'])
def add_router():
    router = request.form.get('router')
    if router:
        graph.add_node(router)
    return visualize()  # Updated to render the graph immediately

# Function to Remove Router (Node)
@app.route('/remove_router', methods=['POST'])
def remove_router():
    router = request.form.get('router').strip()
    if router in graph.nodes:
        graph.remove_node(router)
        return visualize()  # Ensures UI updates immediately after router removal
    return jsonify({'error': f'Router {router} not found.'})

# Function to Add Link (Edge)
@app.route('/add_link', methods=['POST'])
def add_link():
    src = request.form.get('src')
    dest = request.form.get('dest')
    cost = request.form.get('cost')
    if src and dest and cost:
        graph.add_edge(src, dest, weight=int(cost))
    return visualize()  # Updated to render the graph immediately

# Function to Modify Link (Change Cost)
@app.route('/modify_link', methods=['POST'])
def modify_link():
    src = request.form.get('src').strip()
    dest = request.form.get('dest').strip()
    new_cost = request.form.get('new_cost').strip()

    if src in graph.nodes and dest in graph.nodes and graph.has_edge(src, dest):
        graph[src][dest]['weight'] = int(new_cost)
        return visualize()  # Update visualization instead of returning JSON
    return jsonify({'error': 'Link not found or invalid nodes.'})

# Function for removing the inserted link
@app.route('/remove_link', methods=['POST'])
def remove_link():
    src = request.form.get('src')
    dest = request.form.get('dest')

    if src in graph.nodes and dest in graph.nodes and graph.has_edge(src, dest):
        graph.remove_edge(src, dest)
        return visualize()  # Updated visualization after link removal
    return jsonify({'error': f'Link between {src} and {dest} not found.'})

# Dijkstra's Algorithm Implementation
@app.route('/dijkstra', methods=['POST'])
def dijkstra():
    source = request.form.get('source').strip()
    if not graph.nodes:
        return jsonify({'error': 'The graph is empty. Add routers first.'})
    if source not in graph.nodes:
        return jsonify({'error': f'Source router {source} not found.'})

    # Dijkstra's Algorithm
    shortest_paths = {node: float('inf') for node in graph.nodes}
    shortest_paths[source] = 0
    pq = [(0, source)]
    # Track the previous node for path reconstruction
    previous_nodes = {source: None}

    while pq:
        current_cost, current_node = heapq.heappop(pq)

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            new_cost = current_cost + weight

            if new_cost < shortest_paths[neighbor]:
                shortest_paths[neighbor] = new_cost
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))

    # Find the shortest path to all nodes
    paths = {}
    for target in graph.nodes:
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = previous_nodes.get(current)
        path.reverse()
        paths[target] = path if path[0] == source else []

    return visualize_with_path(paths)

# Visualizing the Graph with the Shortest Path Highlighted
def add_router_icons(pos, ax, paths=None):
    icon_path = os.path.join('static', 'wifi.png')
    if not os.path.exists(icon_path):
        return

    image = plt.imread(icon_path)
    imagebox = OffsetImage(image, zoom=0.1)

    for node, (x, y) in pos.items():
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
        ax.text(x, y - 0.08, node, fontsize=9, ha='center')

def visualize():
    plt.clf()
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, ax=ax, node_size=0)  # Hide default nodes

    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax)

    add_router_icons(pos, ax)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return render_template('index.html', graph_image=encoded_img)

def visualize_with_path(paths):
    plt.clf()
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, ax=ax, node_size=0)  # Hide default nodes

    # Highlighting shortest paths
    for path in paths.values():
        if len(path) > 1:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2.5, edge_color='red', ax=ax)

    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax)

    add_router_icons(pos, ax, paths)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return render_template('index.html', graph_image=encoded_img)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
