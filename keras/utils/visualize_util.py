import os
import pdb

from ..layers.wrappers import Wrapper
from ..models import Sequential

try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
if not pydot.find_graphviz():
    raise RuntimeError('Failed to import pydot. You must install pydot'
                       ' and graphviz for `pydotprint` to work.')

def graph_node(layers, dot, model, show_shapes=False, show_layer_names=False):
    for l, layer in enumerate(layers):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
	if class_name in [ "Dropout", "BatchNormalization", "Activation", "Flatten"]:
	    continue
	if class_name == "Model":
	    layer.layers[0].inbound_nodes=layer.inbound_nodes
	    graph_node(layer.layers, dot, layer, show_shapes=show_shapes)
	    continue
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name.split("(")[1][:-1])
        else:
            label = class_name.split("(")[-1].split(")")[0]

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
		if len(layer.output_shape)>2:
                    outputlabels = str(layer.output_shape[2:])
	 	else:
		    outputlabels = str(layer.output_shape[1])

            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
		if len(layer.input_shape)>2:
                    inputlabels = str(layer.input_shape[2:])
		else:
		    inputlabels = str(layer.input_shape[1])
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

	if "Convolution" in class_name:
  	    color = "red"
	elif "Pooling" in class_name:
	    color = "blue"
	else: color = "green" 

        node = pydot.Node(layer_id, label=label, color=color)
        dot.add_node(node)

def edges(layers, dot, model):
    for l, layer in enumerate(layers):
        class_name = layer.__class__.__name__
	if class_name in [ "Dropout", "BatchNormalization", "Activation", "Flatten"]:
	    if l<(len(layers)-1):
	    	layers[l+1].inbound_nodes=layer.inbound_nodes
	    	continue
	layer_id = str(id(layer))
	if class_name == "Model":
	    #layer.layers[0].inbound_nodes=layer.inbound_nodes
	    layer_id, bottom = edges(layer.layers, dot, layer)
	    for i, inbound_layer in enumerate(layer.outbound_nodes[0].inbound_layers):
		if str(id(layer)) == str(id(inbound_layer)):
		    layer.outbound_nodes[0].inbound_layers[i] = bottom
	#pdb.set_trace()
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id, height=0.2, color="black"))
    return [str(id(layers[0])), layers[-1]]

def model_to_dot(model, show_shapes=False, show_layer_names=False):
    dot = pydot.Dot(format='png')
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    
    graph_node(layers, dot, model, show_shapes=show_shapes, show_layer_names=show_layer_names)

    # Connect nodes with edges.
    edges(layers, dot, model)

    return dot

def model_to_dot2(model, show_shapes=False, show_layer_names=True, layer_names = [], keep_layers = []):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

def plot(model, to_file='model.png', show_shapes=False, show_layer_names=False):
    dot = model_to_dot(model, show_shapes, show_layer_names)
    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for l, layer in enumerate(layers[keep_layers]):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer_names[l]
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for l, layer in enumerate(layers[keep_layers]):
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


