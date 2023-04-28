<html>

<head>
    <title>Pympler - {{title}}</title>
    <link rel="stylesheet" type="text/css" href="/static/style.css">
    <script src="http://code.jquery.com/jquery-1.10.1.min.js" type="text/javascript"></script>
</head>

%navbar = [
%    ("overview", "/", ""),
%    ("|", "", ""),
%    ("process", "/process", ""), 
%    ("|", "", ""),
%    ("tracked objects", "/tracker", ""),
%    ("|", "", ""),
%    ("garbage", "/garbage", ""),
%    ("help", "/help", "right"),]

<body>
<div class="related">
    <ul>
        %for link, href, cls in navbar:
            <li class="{{cls}}">
                %if bool(href):
                    <a href="{{href}}"><span>{{link}}</span></a>
                %else:
                    <span>{{link}}</span>
                %end
            </li>
        %end
    </ul>
</div>
<div class="document">
<div class="documentwrapper">
<div class="bodywrapper">
<div class="body">
