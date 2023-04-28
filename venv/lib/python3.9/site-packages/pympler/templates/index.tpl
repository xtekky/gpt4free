%include('header', category='Overview', title='Overview')

%from pympler.util.stringutils import pp

<h1>Python application memory profile</h1>

<h2>Process overview</h2>

<table class="tdata">
    <tbody>
    <tr>
        <th>Virtual size:</th>
        <td class="num">{{pp(processinfo.vsz)}}</td>
    </tr>
    <tr>
        <th>Physical memory size:</th>
        <td class="num">{{pp(processinfo.rss)}}</td>
    </tr>
    <tr>
        <th>Major pagefaults:</th>
        <td class="num">{{processinfo.pagefaults}}</td>
    </tr>
    </tbody>
</table>

%include('footer')
