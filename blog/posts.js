---
layout: none
permalink: /posts.js
---

window.__BLOG_POSTS__ = [
{% for post in site.posts %}

  {% assign raw = post.content %}

  {% if raw contains '<div class="lang-en"' %}
    {% assign inside = raw | split: '<div class="lang-en" markdown="1">' | last | split: '</div>' | first %}
  {% else %}
    {% assign inside = "" %}
  {% endif %}

  {
    title: {{ post.title_en | default: post.title | jsonify }},
    url: {{ post.url | jsonify }},
    date: {{ post.date | date: "%Y-%m-%d" | jsonify }},
    categories: {{ post.categories | jsonify }},
    excerpt: {{ inside | strip_html | strip_newlines | truncate: 220 | jsonify }}
  }{% unless forloop.last %},{% endunless %}

{% endfor %}
];