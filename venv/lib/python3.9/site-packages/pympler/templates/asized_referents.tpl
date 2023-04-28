%for ref in referents:
    <div class="referents">
        <span class="local_name">{{ref.name}}</span>
        <span class="local_size">{{ref.size}}</span>
        %if ref.refs:
            %include('asized_referents', referents=ref.refs)
        %end
    </div>
%end
