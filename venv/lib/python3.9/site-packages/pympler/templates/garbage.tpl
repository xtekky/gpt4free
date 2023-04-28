%include('header', category='Garbage', title='Garbage')

<h1>Garbage - Cycle {{index}}</h1>
<table class="tdata" width="100%">
    <thead>
        <tr>
            <th>id</th>
            <th class="num">size</th>
            <th>type</th>
            <th>representation</th>
        </tr>
    </thead>
    <tbody>
    %for o in objects:
        <tr>
            <td>{{'0x%08x' % o.id}}</td>
            <td class="num">{{o.size}}</td>
            <td>{{o.type}}</td>
            <td>{{o.str}}</td>
        </tr>
    %end
    </tbody>
</table>

<h2>Reference graph</h2>

<img src="/garbage/graph/{{index}}"/>

<h2>Reduced reference graph (cycles only)</h2>

<img src="/garbage/graph/{{index}}?reduce=1"/>

%include('footer')
