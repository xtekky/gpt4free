%include('header', category='Tracker', title=clsname)

%from pympler.util.stringutils import pp, pp_timestamp

<h1>{{clsname}}</h1>

%sizes = [tobj.get_max_size() for tobj in stats.index[clsname]]

<p>{{len(stats.index[clsname])}} instances of {{clsname}} were registered. The
average size is {{pp(sum(sizes)/len(sizes))}}, the minimal size is
{{pp(min(sizes))}}, the maximum size is {{pp(max(sizes))}}.</p>

<h2>Coalesced Referents per Snapshot</h2>

%for snapshot in stats.snapshots:
    %if clsname in snapshot.classes:
        %merged = snapshot.classes[clsname]['merged']
        <h3>Snapshot: {{snapshot.desc}}</h3>
        <p>{{pp(merged.size)}} occupied by instances of class {{clsname}}</p>
        %if merged.refs:
            %include('asized_referents', referents=merged.refs)
        %else:
            <p>No per-referent sizes recorded.</p>
        %end
    %end
%end

<h2>Instances</h2>

%for tobj in stats.index[clsname]:
    <table class="tdata" width="100%" rules="rows">
        <tr>
            <th width="140px">Instance</th>
            <td>{{tobj.name}} at {{'0x%08x' % tobj.id}}</td>
        </tr>
        %if tobj.repr:
            <tr>
                <th>Representation</th>
                <td>{{tobj.repr}}&nbsp;</td>
            </tr>
        %end
        <tr>
            <th>Lifetime</th>
            <td>{{pp_timestamp(tobj.birth)}} - {{pp_timestamp(tobj.death)}}</td>
        </tr>
        %if getattr(tobj, 'trace'):
            <tr>
                <th>Instantiation</th>
                <td>
                    % # <div class="stacktrace">
                        %for frame in tobj.trace:
                            <div class="stackframe">
                                <span class="filename">{{frame[0]}}</span>
                                <span class="lineno">{{frame[1]}}</span>
                                <span class="function">{{frame[2]}}</span>
                                <div class="context">{{frame[3][0].strip()}}</div>
                            </div>
                        %end
                    % # </div>
                </td>
            </tr>
        %end
        %for (timestamp, size) in tobj.snapshots:
            <tr>
            <td>{{pp_timestamp(timestamp)}}</td>
            %if not size.refs:
                <td>{{pp(size.size)}}</td>
            %else:
                <td>
                    {{pp(size.size)}}
                    %include('asized_referents', referents=size.refs)
                </td>
            %end
            </tr>
        %end
    </table>
%end

%include('footer')
