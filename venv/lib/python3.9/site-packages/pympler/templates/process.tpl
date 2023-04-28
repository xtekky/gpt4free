%include('header', category='Process', title='Process Information')
%from pympler.util.stringutils import pp

<h1>Process information</h1>

<table class="tdata">
    <tbody>
    <tr>
        <th>Virtual size:</th>
        <td class="num">{{pp(info.vsz)}}</td>
    </tr>
    <tr>
        <th>Physical memory size:</th>
        <td class="num">{{pp(info.rss)}}</td>
    </tr>
    <tr>
        <th>Major pagefaults:</th>
        <td class="num">{{info.pagefaults}}</td>
    </tr>
    %for key, value in info.os_specific:
        <tr>
            <th>{{key}}:</th>
            <td class="num">{{value}}</td>
        </tr>
    %end
    </tbody>
</table>

<h2>Thread information</h2>

<table class="tdata">
    <tbody>
    <tr>
        <th>ID</th>
        <th>Name</th>
        <th>Daemon</th>
    </tr>
    %for tinfo in threads:
        <tr>
            <td>{{tinfo.ident}}</td>
            <td>{{tinfo.name}}</td>
            <td>{{tinfo.daemon}}</td>
        </tr>
    %end
    </tbody>
</table>

<h2>Thread stacks</h2>

<div class="stacks">
    %for tinfo in threads:
        <div class="stacktrace" id="{{tinfo.ident}}">
            <a class="show_traceback" href="#">Traceback for thread {{tinfo.name}}</a>
        </div>
    %end
</div>

<script type="text/javascript">
    $(".show_traceback").click(function() {
        var tid = $(this).parent().attr("id");
        $.get("/traceback/"+tid, function(data) {
            $("#"+tid).replaceWith(data);
        });
        return false;
    });
    $(".stacks").delegate(".expand_local", "click", function() {
        var oid = $(this).attr("id");
        $.get("/objects/"+oid, function(data) {
            $("#"+oid).replaceWith(data);
        });
        return false;
    });
    $(".stacks").delegate(".expand_ref", "click", function() {
        var node_id = $(this).attr("id");
        var oid = node_id.split("_")[0];
        $.get("/objects/"+oid, function(data) {
            $("#children_"+node_id).append(data);
        });
        $(this).removeClass("expand_ref").addClass("toggle_ref");
        return false;
    });
    $(".stacks").delegate(".toggle_ref", "click", function() {
        var node_id = $(this).attr("id");
        $("#children_"+node_id).toggle();
        return false;
    });
</script>

%include('footer')
