---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<p class="display-4">Lecture notes</p>
<div class="container p-0">
{% assign tags =  site.content | map: 'tags' | uniq %}
{% for tag in tags %}
  <dl class="row">
  	<dt class="col-sm-4 col-12">{{ tag }}</dt>
  	<dd class="col-sm-6 col-12">
  	{% for note in site.content %}
      {% if note.tags contains tag %}
      <p><a href="{{ site.baseurl }}{{ note.url }}">{{ note.title }}</a></p>
      {% endif %}
  	{% endfor %}
  </dd>
  </dl>
{% endfor %}
</div>

<p class="display-4">Projects</p>