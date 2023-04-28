<div class="stacktrace">
    <strong>Stacktrace for thread {{threadid}}</strong>
    %for frame in stack:
        <div class="stackframe">
            <span class="filename">{{frame[1]}}</span>
            <span class="lineno">{{frame[2]}}</span>
            <span class="function">{{frame[3]}}</span>
            %if frame[4]:
                %context = frame[4]
                <div class="context">
                    %highlight = len(context) / 2
                    %for idx, line in enumerate(frame[4]):
                        %hl = (idx == highlight) and "highlighted" or ""
                        %if line.strip():
                            <div class="{{hl}}" style="padding-left:{{len(line)-len(line.lstrip())}}em" width="100%">
                                {{line.strip()}}
                            </div>
                        %end
                    %end
                </div>
            %end
            <div class="local">
                <a class="expand_local" id="{{frame[0]}}"  href="#">Show locals</a>
            </div>
        </div>
    %end
    %if not stack:
        Cannot retrieve stacktrace for thread {{threadid}}.
    %end
</div>
