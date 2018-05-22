"""Graph module provides APIs for layer-graph."""


class Graph(object):

    class Node(object):

        def __init__(self, layer):
            self.name = layer.name
            self.out_shape = layer.outputs.shape
            self.type = layer.__class__.__name__
            ps = set(layer.all_params)
            for prev in layer.prev_layers:
                ps = ps - set(prev.all_params)
            self.params = list(ps)

        def gen_dot(self):
            lines = [
                'name: %s' % self.name,
                'out_shape: %s' % self.out_shape,
                'type: %s' % self.type,
            ]
            for i, p in enumerate(self.params):
                lines.append('param%d: %s %s' % (i, p.name, p.shape))
            return ''.join(l + '\\l' for l in lines)

    class Edge(object):

        def __init__(self, src, dst):
            self.src = src
            self.dst = dst

    def __init__(self):
        self.nodes = dict()
        self.edges = []

    def add_node(self, layer):
        self.nodes[layer.name] = Graph.Node(layer)

    def get_or_add_node(self, layer):
        name = layer.name
        if name not in self.nodes:
            self.add_node(layer)
        return self.nodes[name]

    def add_edge(self, p, q):
        e = Graph.Edge(p, q)
        self.edges.append(e)

    def gen_dot(self, f):
        f.write('digraph {\n')
        # nodes
        for n in self.nodes.values():
            f.write('\t"%s" [shape="box", label="%s"];\n' % (n.name, n.gen_dot()))
        f.write('\n')
        # edges
        for e in self.edges:
            f.write('\t "%s" -> "%s";\n' % (e.src.name, e.dst.name))
        f.write('}\n')


def build_graph(net):
    g = Graph()
    visited = dict()

    def visit(net):
        if net in visited:
            return
        visited[net] = True
        g.add_node(net)
        for prev in net.prev_layers:
            p = g.get_or_add_node(prev)
            q = g.get_or_add_node(net)
            g.add_edge(p, q)
            visit(prev)

    visit(net)
    return g
