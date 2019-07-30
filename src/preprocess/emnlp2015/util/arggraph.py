#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import networkx as nx
from lxml import etree
from queue import Queue
from textwrap import wrap
from pydot import graph_from_dot_data


# http://stackoverflow.com/a/2669120
def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


class ArgGraphException(nx.NetworkXException):
    ''' A class for exceptions raised in the handling of argumentation
        graphs '''


class ArgGraph (nx.DiGraph):

    def add_edu(self, edu_id, edu_txt):
        self.add_node(edu_id, type="edu", text=edu_txt)

    def add_edu_joint(self, edu_id):
        self.add_node(edu_id, type="joint")

    def add_adu(self, adu_id, adu_role):
        self.add_node(adu_id, type="adu", role=adu_role)

    def add_seg_edge(self, edge_src, edge_trg):
        # print "edge (seg): ", edge_src, edge_trg
        self.add_edge(edge_src, edge_trg, type="seg")

    def add_edge_with_relation_node(self, edge_id, edge_src, edge_trg,
                                    edge_type):
        # print "edge (%s): %s ==[%s]==> %s" % (edge_type, edge_src, edge_id,
        #                                       edge_trg)
        self.add_node(edge_id, type="rel")
        self.add_edge(edge_src, edge_id, type="src")
        self.add_edge(edge_id, edge_trg, type=edge_type)

    def load_from_xml(self, filename):
        xml = etree.parse(filename)

        # graph id
        text_id = xml.xpath('/arggraph')[0].get('id')
        self.graph['id'] = text_id

        # add all EDU
        for elm in xml.xpath('/arggraph/edu'):
            self.add_edu(elm.get('id'), elm.text)
        # add all EDU-JOINS
        for elm in xml.xpath('/arggraph/joint'):
            self.add_edu_joint(elm.get('id'))
        # add all ADU
        for elm in xml.xpath('/arggraph/adu'):
            self.add_adu(elm.get('id'), elm.get('type'))

        # add all edges
        q = Queue()
        for elm in xml.xpath('/arggraph/edge'):
            q.put(elm)
        while not q.empty():
            # TODO: queue processing might not end for input elements with
            #       malformed targets, cyclic relations
            elm = q.get()

            edge_src = elm.get('src')
            if edge_src not in self.nodes():
                print("Error: source unknown\n", etree.tostring(elm))

            edge_trg = elm.get('trg')
            if edge_trg not in self.nodes():
                # target node (of 'und' or 'add' relations) not there yet.
                # postpone to later
                q.put(elm)
                continue

            edge_type = elm.get('type')
            edge_id = elm.get('id')
            if edge_type == 'seg':
                src_trg_type_pair = (self.node[edge_src]['type'],
                                     self.node[edge_trg]['type'])
                if src_trg_type_pair in [('edu', 'adu'), ('edu', 'joint'),
                                         ('joint', 'adu')]:
                    self.add_seg_edge(edge_src, edge_trg)
                else:
                    print("Error: malformed segmentation edge\n", \
                        etree.tostring(elm))

            elif edge_type in ['sup', 'exa', 'reb']:
                if (self.node[edge_src]['type'] == 'adu' and
                        self.node[edge_trg]['type'] == 'adu'):
                    self.add_edge_with_relation_node(edge_id, edge_src,
                                                     edge_trg, edge_type)
                else:
                    print("Error: malformed direct edge\n", etree.tostring(elm))

            elif edge_type == 'und':
                if (self.node[edge_src]['type'] == 'adu' and
                        self.node[edge_trg]['type'] == 'rel'):
                    self.add_edge_with_relation_node(edge_id, edge_src,
                                                     edge_trg, edge_type)
                else:
                    print(("Error: malformed undercutting edge\n",
                           etree.tostring(elm)))

            elif edge_type == 'add':
                if (self.node[edge_src]['type'] == 'adu' and
                        self.node[edge_trg]['type'] == 'rel'):
                    self.add_edge(elm.get('src'), elm.get('trg'), type='src')
                else:
                    print("Error: malformed adding edge\n", etree.tostring(elm))

            else:
                print("Error: unknown edge type\n", etree.tostring(elm))

        # update adu short names
        self.update_adu_labels()

    def update_adu_labels(self):
        # first label all edus
        for edu_node in [i for i, d in self.nodes(data=True)
                         if d['type'] == 'edu']:
            self.node[edu_node]['nr-label'] = edu_node.replace('e', '')

        # then all joints
        for joint_node in [i for i, d in self.nodes(data=True)
                           if d['type'] == 'joint']:
            label = '+'.join(sorted_nicely(
                [self.node[i]['nr-label']
                 for i in self.predecessor_by_edge_type(joint_node, 'seg')]
            ))
            self.node[joint_node]['nr-label'] = label

        # then all adu
        for adu_node in [i for i, d in self.nodes(data=True)
                         if d['type'] == 'adu']:
            label = '='.join(sorted_nicely(
                [self.node[i]['nr-label']
                 for i in self.predecessor_by_edge_type(adu_node, 'seg')]
            ))
            self.node[adu_node]['nr-label'] = label

    def predecessors_with_node_type(self, node, node_type):
        return [i for i in self.predecessors(node)
                if 'type' in self.node[i] and self.node[i]['type'] == type]

    def predecessor_by_edge_type(self, node, edge_type):
        return [src for src, trg, d in self.edges(data=True)
                if trg == node and 'type' in d and d['type'] == edge_type]

    def get_relation_node_free_graph(self):
        if nx.is_strongly_connected(self):
            raise ArgGraphException(('Cannot produce relation node free graph.'
                                     'Arggraph contains cycles.'))
        if False in [self.out_degree(node) <= 1 for node in self.nodes()]:
            raise ArgGraphException(('Cannot produce relation node free graph.'
                                    'Nodes with multiple outgoing edges.'))

        a = ArgGraph(self)

        if ('relation-node-free' in a.graph and
                a.graph['relation-node-free'] == True):
            return a

        # reduce multi-source relations to adu.addsource->adu
        for rel_node in [node for node, d in a.nodes(data=True)
                         if a.out_degree(node) >= 1 and d['type'] == 'rel']:
            sources = sorted_nicely(
                [source for source in a.predecessors(rel_node)
                 if a.node[source]['type'] == 'adu']
            )
            for source in sources[1:]:
                a.remove_edge(source, rel_node)
                a.add_edge(source, sources[0], type="add")

        # first reduce rel->rel
        remove_nodes = []
        remove_edges = []
        for (src, trg, d) in list(a.edges(data=True)):
            if a.node[src]['type'] == 'rel' and a.node[trg]['type'] == 'rel':
                src_pre = a.predecessor_by_edge_type(src, 'src')[0]
                trg_pre = a.predecessor_by_edge_type(trg, 'src')[0]
                a.remove_edge(src, trg)
                a.add_edge(src_pre, trg_pre, type=d['type'])
                remove_edges.append((src_pre, src, ))
                remove_nodes.append(src)

        for src, trg in remove_edges:
            a.remove_edge(src, trg)
        for node in remove_nodes:
            a.remove_node(node)

        # then reduce rel->adu (remaining relnodes)
        for (src, trg, d) in list(a.edges(data=True)):
            if a.node[src]['type'] == 'rel' and a.node[trg]['type'] == 'adu':
                src_pre = list(a.predecessors(src))[0]
                a.add_edge(src_pre, trg, type=d['type'])
                a.remove_edge(src_pre, src)
                a.remove_edge(src, trg)
                a.remove_node(src)

        a.graph['relation-node-free'] = True

        return a

    def export_to_dot(self, edu_cluster=False):
        queries_nodes = {
            'edu': lambda d: d['type'] == 'edu',
            'joint': lambda d: d['type'] == 'joint',
            'pro': lambda d: d['type'] == 'adu' and d['role'] == 'pro',
            'opp': lambda d: d['type'] == 'adu' and d['role'] == 'opp',
            'rel': lambda d: d['type'] == 'rel',
        }

        styles_nodes = {
            'edu': ('[shape=box, style=filled, color="#aaaaaa", '
                    'fillcolor="#dddddd", fontsize=10, width=2.5];'
                    '\nstyle=invis;'),
            'joint': ('[shape=box, style=filled, color="#aaaaaa", '
                      'fillcolor="#dddddd"];'),
            'pro': '[shape=oval];',
            'opp': '[shape=rect];',
            'rel': ('[shape=octagon, style=filled, color="#aaaaaa", '
                    'fixedsize=true, width=0.3, height=0.3, '
                    'fillcolor="#FFF8DC", fontsize=10];'),
        }

        styles_edges = {
            'seg': '[weight=1, arrowhead=none, color="#aaaaaa"];',
            'src': '[weight=1, arrowhead=none];',
            'sup': '[weight=1, arrowhead=open];',
            'exa': '[weight=1, arrowhead=open, style=dashed];',
            'reb': '[weight=1, arrowhead=dot];',
            'und': '[weight=1, arrowhead=box];',
            'add': '[weight=1, arrowhead=empty];',
        }

        # TODO:
        # When plotting the graph TB (thus linearizing EDUS left to right)
        # crossing edges work.
        # When plotting the graph LR (thus linearizing EDUS top down)
        # crossing edges don't work.

        # template_graph = u'digraph G {\n// %s\nrankdir=LR\n%s}' # name content  # noqa
        template_graph = 'digraph G {\n// %s\n%s}'  # name content
        template_subgraph = 'subgraph %s {\n%s\n}'  # name content
        template_node_label = '%s [label="%s"];'  # id label

        edu_content = ''

        # first edus nodes
        node_type = 'edu'
        data = 'node ' + styles_nodes[node_type]
        data += '\nrank=same;'
        data += '\nrankdir=TB;'
        nodes = sorted_nicely([i for (i, d) in self.nodes(data=True)
                               if queries_nodes[node_type](d)])
        for i in nodes:
            text = self.node[i]['text'].replace('"', '\'\'')
            wrapped_text = wrap('[%s] ' % i + text, width=30)
            data += '\n' + template_node_label % (i, '\\n'.join(wrapped_text))

        # add invisible edges from edu to edu to enforce linearity in the graph
        # template_linearity_subgraph = '\nsubgraph linearity {\nedge [weight=8, color="#ffcccc"];\n%s}'  # noqa
        template_linearity_subgraph = '\nsubgraph linearity {\nedge [weight=8, style=invis];\n%s}'  # noqa
        edges = ''
        for src, trg in zip(nodes, nodes[1:]):
            edges += '%s -> %s\n' % (src, trg)
        linearity_subgraph = template_linearity_subgraph % edges

        if edu_cluster:
            subgraph = template_subgraph % ('cluster_' + node_type,
                                            data + linearity_subgraph)
        else:
            subgraph = template_subgraph % (node_type,
                                            data + linearity_subgraph)
        edu_content += '\n' + subgraph

        graph_content = ''

        # remaining nodes
        for node_type in styles_nodes:
            if node_type == 'edu':
                continue
            data = ''
            data += 'node ' + styles_nodes[node_type]
            nodes = [(i, d) for (i, d) in self.nodes(data=True)
                     if queries_nodes[node_type](d)]
            for i, d in nodes:
                node_label = i
                if 'nr-label' in self.node[i]:
                    node_label = self.node[i]['nr-label']
                data += '\n' + template_node_label % (i, node_label)
            graph_content += '\n' + template_subgraph % (node_type, data)

        # edges
        for edge_type in styles_edges:
            data = ''
            data += 'edge ' + styles_edges[edge_type]
            edges = [(src, trg, d) for (src, trg, d) in self.edges(data=True)
                     if d['type'] == edge_type]
            for src, trg, _ in edges:
                data += '\n' + src + ' -> ' + trg
            graph_content += '\n' + template_subgraph % (edge_type, data)

        content = (edu_content + '\n' +
                   template_subgraph % ('graph_rest', graph_content))

        dot = template_graph % ("automatically generated", content)
        return dot

    def show_in_ipynb(self, edu_cluster=False):
        from IPython.display import Image
        dot = self.export_to_dot(edu_cluster=edu_cluster).encode('utf-8')
        return Image(graph_from_dot_data(dot).create_png())

    def render_as_dot(self, edu_cluster=False):
        dot = self.export_to_dot(edu_cluster=edu_cluster)
        dot_utf8 = dot.encode('utf-8')
        return dot_utf8

    def render_as_png(self, filename, edu_cluster=False):
        dot_utf8 = self.render_as_dot(edu_cluster=edu_cluster)
        dot_graph = graph_from_dot_data(dot_utf8)
        dot_graph.write_png(filename)

    def render_as_pdf(self, filename, edu_cluster=False):
        dot_utf8 = self.render_as_dot(edu_cluster=edu_cluster)
        dot_graph = graph_from_dot_data(dot_utf8)
        dot_graph.write_pdf(filename)

    def get_edus(self):
        return {i: d['text'] for (i, d) in self.nodes(data=True)
                if d['type'] == 'edu'}

    def get_adus(self):
        return set([i for (i, d) in self.nodes(data=True)
                    if d['type'] == 'adu'])

    def get_segmented_text(self):
        edus = self.get_edus()
        return [edus[i] for i in sorted_nicely(list(edus.keys()))]

    def get_unsegmented_text(self):
        return ' '.join(self.get_segmented_text())

    def get_adu_adu_relations(self):
        # make sure to extract relations from a relation node free graph
        if ('relation-node-free' in self.graph and
                self.graph['relation-node-free'] == True):
            g = self
        else:
            g = self.get_relation_node_free_graph()

        # get all adu-adu relations
        return [(src, trg, d['type'])
                for src, trg, d in g.edges(data=True)
                if (g.node[src]['type'] == 'adu' and
                    g.node[trg]['type'] == 'adu')]

    def get_adus_as_dependencies(self):
        # make sure to extract relations from a relation node free graph
        if ('relation-node-free' in self.graph and
                self.graph['relation-node-free'] == True):
            g = self
        else:
            g = self.get_relation_node_free_graph()
        # get central claim and adu_adu relations
        r = [(g.get_central_claim(), 'a0', 'ROOT')]
        r.extend(g.get_adu_adu_relations())
        return sorted(r)

    def get_adu_role(self, adu):
        return self.node[adu]['role']

    def get_adu_functions(self, adu):  # todo central claim
        if ('relation-node-free' in self.graph and
                self.graph['relation-node-free'] == True):
            return self.edges(adu, data=True)
        else:
            # outgoing arcs go to relation nodes first, we want the arc _from_
            # that relation node
            out = self.edges(adu)
            if len(out) == 0:
                return []
            else:
                relnode = self.edges(adu)[0][1]
                return self.edges(relnode, data=True)

    def get_central_claim(self):
        _outdegree, ccnode = min([(self.out_degree(n), n)
                                  for n in self.nodes()])
        return ccnode

    def get_role_type_labels(self):
        aar = self.get_adu_adu_relations()
        return {src: '%s+%s' % (self.node[src]['role'], dtype)
                for src, _trg, dtype in aar}
