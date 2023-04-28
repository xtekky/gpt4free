%from random import randint
%for name, (ref, type_name, obj_repr, size) in referents.items():
    %ref = "%s_%s" % (ref, randint(0, 65535))
    <div class="referents">
        <a class="expand_ref" id="{{ref}}" href="#">
            <span class="local_name">{{name}}</span>
            <span class="local_type">{{type_name}}</span>
            <span class="local_size">{{size}}</span>
            <span class="local_value">{{obj_repr}}</span>
        </a>
        <span id="children_{{ref}}"/>
    </div>
%end            
