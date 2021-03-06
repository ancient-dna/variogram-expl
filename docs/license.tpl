{%- extends 'basic.tpl' -%}

{%- block header -%}
{{ super() }}
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="ipynb_website:version" content="0.9.2" />
<meta name="viewport" content="width=device-width, initial-scale=1" />

<link rel="stylesheet" type="text/css" href="../css/jt.css">

<link rel="stylesheet" type="text/css" href="../css/toc2.css">

<link href="../site_libs/jqueryui-1.11.4/jquery-ui.css">
<link rel="stylesheet" href="../site_libs/bootstrap-3.3.5/css/cosmo.min.css" rel="stylesheet" />
<link rel="stylesheet" href="../site_libs/font-awesome-4.5.0/css/font-awesome.min.css" rel="stylesheet" />
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.9.1/jquery-ui.min.js"></script>
<script src="../site_libs/bootstrap-3.3.5/js/bootstrap.min.js"></script>
<script src="../site_libs/bootstrap-3.3.5/shim/html5shiv.min.js"></script>
<script src="../site_libs/bootstrap-3.3.5/shim/respond.min.js"></script>

<link rel="stylesheet"
      href="../site_libs/highlight/textmate.css"
      type="text/css" />

<script src="../site_libs/highlight/highlight.js"></script>
<script type="text/javascript">
if (window.hljs && document.readyState && document.readyState === "complete") {
   window.setTimeout(function() {
      hljs.initHighlighting();
   }, 0);
}
</script>

<script src="../js/toc2.js"></script>
<script src="../js/docs.js"></script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>
<script>
    MathJax.Hub.Config({
        extensions: ["tex2jax.js"],
        jax: ["input/TeX", "output/HTML-CSS"],
        tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
        processEscapes: true
        },
        "HTML-CSS": {
            preferredFont: "TeX",
            availableFonts: ["TeX"],
            styles: {
                scale: 110,
                ".MathJax_Display": {
                    "font-size": "110%",
                }
            }
        }
    });
</script>

<script>
// manage active state of menu based on current page
$(document).ready(function () {
  // active menu anchor
  href = window.location.pathname
  href = href.substr(href.lastIndexOf('/') + 1)
  if (href === "")
    href = "index.html";
  var menuAnchor = $('a[href="' + href + '"]');
  // mark it active
  menuAnchor.parent().addClass('active');
  // if it's got a parent navbar menu mark it active as well
  menuAnchor.closest('li.dropdown').addClass('active');
});
</script>
<div class="container-fluid main-container">
<!-- tabsets -->
<script src="../site_libs/navigation-1.1/tabsets.js"></script>
<script>
$(document).ready(function () {
  window.buildTabsets("TOC");
});
</script>



<title>Variogram exploration</title>

<style type = "text/css">
body {
  
  padding-top: 66px;
  padding-bottom: 40px;
}
</style>
</head>

<body>
<div tabindex="-1" id="notebook" class="border-box-sizing">
<div class="container" id="notebook-container">

<!-- code folding -->

<div class="navbar navbar-default  navbar-fixed-top" role="navigation">
  <div class="container">
    <div class="navbar-header">
      <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar">
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
      </button>
      <a class="navbar-brand" href="../index.html">Variogram exploration</a>
    </div>
    <div id="navbar" class="navbar-collapse collapse">
      <ul class="nav navbar-nav">
        
<li>
  <a href="../license.html">License</a>
</li>
        
      </ul>
        
<ul class="nav navbar-nav navbar-right">
<li>
   <a href="https://github.com/ancient-dna/variogram-expl"> source </a>
</li>
</ul>
        
      </div><!--/.nav-collapse -->
  </div><!--/.container -->
</div><!--/.navbar -->

{%- endblock header -%}

{% block footer %}
<hr>

</div>
</div>
</body>
</html>
{% endblock %}