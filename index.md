---
layout: default
---

<body>
  <div class="index-wrapper">
    <div class="aside">
       <div class="info-card">
        <h1>Wenlong</h1>
        <a href="https://www.linkedin.com/in/cao-wenlong-7b691262/" target="_blank"><img src="https://www.linkedin.com/favicon.ico" alt="" width="25"/></a>
        <a href="https://www.quora.com/profile/Charles-Cao-7" target="_blank"><img src="https://www.quora.com/favicon.ico" alt="" width="22"/></a>

      </div>
      <div id="particles-js"></div>
    </div>

    <div class="index-content">
      <ul class="artical-list">
        {% for post in site.categories.blog %}
        <li>
          <a href="{{ post.url }}" class="title">{{ post.title }}</a>
          <div class="title-desc">{{ post.description }}</div>
        </li>
        {% endfor %}
      </ul>
    </div>
  </div>
</body>