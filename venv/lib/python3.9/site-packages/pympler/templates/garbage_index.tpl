%include('header', category='Garbage', title='Garbage')
<h1>Garbage - Overview</h1>

<p>This page gives an overview of all objects that would have been
deleted if those weren't holding circular references to each other
(e.g. in a doubly linked list).</p>

%if len(graphs):
    <p>Click on the reference graph titles below to show the objects
    contained in the respective cycle. If you have <a
    href="http://www.graphviz.org">graphviz</a> installed, you will
    also see a visualisation of the reference cycle.</p>

    <p>{{len(graphs)}} reference cycles:</p>

    <table class="tdata">
        <thead>
            <tr>
                <th>Reference graph</th>
                <th># objects</th>
                <th># cycle objects</th>
                <th>Total size</th>
            </tr>
        </thead>
        <tbody>
        %for graph in graphs:
            <tr>
                <td><a href="/garbage/{{graph.index}}">Cycle {{graph.index}}</a></td>
                <td class="num">{{len(graph.metadata)}}</td>
                <td class="num">{{graph.num_in_cycles}}</td>
                <td class="num">{{graph.total_size}}</td>
            </tr>
        %end
        </tbody>
    </table>

%else:
    <p>No reference cycles detected.</p>
%end

%include('footer')
