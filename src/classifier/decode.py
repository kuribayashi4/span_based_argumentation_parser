# This code is the modifier version of https://github.com/mlbright/edmonds/blob/master/edmonds/edmonds.py
import chainer
import chainer.functions as chaFunc
import numpy as np


def _load(arcs,weights):
    g = {}
    for (src,dst) in arcs:
        if src in g:
            g[src][dst] = weights[(src,dst)]
        else:
            g[src] = { dst : weights[(src,dst)] }
    return g


def _reverse(graph):
    r = {}
    for src in graph:
        for (dst,c) in graph[src].items():
            if dst in r:
                r[dst][src] = c
            else:
                r[dst] = { src : c }
    return r


def _getCycle(n, g, visited=None, cycle=None):
    if visited is None:
        visited = set()
    if cycle is None:
        cycle = []
    visited.add(n)
    cycle += [n]
    if n not in g:
        return cycle
    for e in g[n]:
        if e not in visited:
            cycle = _getCycle(e,g,visited,cycle)
    return cycle


def _mergeCycles(cycle,G,RG,g,rg):
    allInEdges = []
    minInternal = None
    minInternalWeight = float("inf")

    # find minimal internal edge weight
    for n in cycle:
        for e in RG[n]:
            if e in cycle:
                if minInternal is None or RG[n][e] < minInternalWeight:
                    minInternal = (n,e)
                    minInternalWeight = RG[n][e]
                    continue
            else:
                allInEdges.append((n,e))        

    # find the incoming edge with minimum modified cost
    minExternal = None
    minModifiedWeight = 0
    for s,t in allInEdges:
        u,v = rg[s].popitem()
        rg[s][u] = v
        w = RG[s][t] - (v - minInternalWeight)
        if minExternal is None or minModifiedWeight > w:
            minExternal = (s,t)
            minModifiedWeight = w

    u,w = rg[minExternal[0]].popitem()
    rem = (minExternal[0],u)
    rg[minExternal[0]].clear()
    if minExternal[1] in rg:
        rg[minExternal[1]][minExternal[0]] = w
    else:
        rg[minExternal[1]] = { minExternal[0] : w }
    if rem[1] in g:
        if rem[0] in g[rem[1]]:
            del g[rem[1]][rem[0]]
    if minExternal[1] in g:
        g[minExternal[1]][minExternal[0]] = w
    else:
        g[minExternal[1]] = { minExternal[0] : w }

# --------------------------------------------------------------------------------- #

def mst(root,G):
    """ The Chu-Lui/Edmond's algorithm
    arguments:
    root - the root of the MST
    G - the graph in which the MST lies
    returns: a graph representation of the MST
    Graph representation is the same as the one found at:
    http://code.activestate.com/recipes/119466/
    Explanation is copied verbatim here:
    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.
    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.
    """

    RG = _reverse(G)
    if root in RG:
        RG[root] = {}
    g = {}

    for n in RG:
        if len(RG[n]) == 0:
            continue
        minimum = float("inf")
        s,d = None,None

        for e in RG[n]:
            if RG[n][e] < minimum:
                minimum = RG[n][e]
                s,d = n,e
        if d in g:
            g[d][s] = RG[s][d] 
        else:
            g[d] = { s : RG[s][d] }
            
    cycles = []
    visited = set()

    for n in g:
        if n not in visited:
            cycle = _getCycle(n,g,visited)
            cycles.append(cycle)
    
    rg = _reverse(g)
    for cycle in cycles:
        if root in cycle:
            continue
        _mergeCycles(cycle, G, RG, g, rg)
    
    return g

# --------------------------------------------------------------------------------- #


def decode(prices, root):

    g = _load(prices, prices)
    h = mst(int(root), g)

    edges = []
    for s in h:
        for t in h[s]:
            edges.append((t, s))
    edges = sorted(edges, key=lambda x: x[0])
    return edges


def decode_mst(ys_link, ts, max_n_spans):

    # ys: [(batchsize*max_n_spans, max_n_spans+1), (all_spans, type_classes)]
    # ts: (batchsize, n_spans, gold_target)
    batchsize = ts.shape[0]
    max_n_spans = ys_link.shape[0] // ts.shape[0]
    candidate_classes = ys_link.shape[1]

    ts = chainer.cuda.to_cpu(ts)
    ys = chainer.cuda.to_cpu(ys_link.data)

    # (batchsize, n_spans, bool)
    mask = ts > -1

    t_len = np.sum(mask, axis=-1).astype(np.int32)
    t_section = np.cumsum(t_len[:-1])

    # (all_spans)
    ts = ts[mask]
    # (all_spans)
    ys = ys[mask.flatten()]
    ts = chaFunc.split_axis(ts, t_section, 0)
    ys = chaFunc.split_axis(ys, t_section, 0)
    ys_mst = []
    for y_matrix in ys:
        costs = {}
        for row, y_scores in enumerate(y_matrix):
            for col, y_score in enumerate(y_scores):
                if y_score.data != -np.inf:
                    # reverse the direction of edges
                    costs[(col, row)] = -float(y_score.data)

        y_tree = decode(costs, max_n_spans)
        ys_mst.append(y_tree)

    # (batchsize, max_n_spans, max_n_targets)
    ys = chaFunc.pad_sequence(ys, length=max_n_spans, padding=-1)

    # (batchsize, max_n_spans, max_n_targets)
    mask = np.zeros(ys.shape).astype(np.bool)

    minus_inf = np.zeros(ys.shape, dtype=np.float32)
    minus_inf.fill(-np.inf)

    for i, y_mst in enumerate(ys_mst):
        for edge in y_mst:
            mask[i, edge[0], edge[1]] = True

    ys_masked = chaFunc.where(mask, ys, minus_inf)

    ys_masked = chaFunc.reshape(ys_masked, (batchsize*max_n_spans, candidate_classes))

    return ys_masked
