
# SBTC - A Complete Beginner's Guide to Django

This repo is to tracking the progress on my efforts to learning something about Django Framework. I'll starting following the Tutorials Series given by Vitor Freitas on his site [SimpleIsBetterThanComplex](http://simpleisbetterthancomplex.com/).

## Table of contents

The Tutorials Seires has sevens (7) parts, named:

- [x] Part 1 - Getting Started
- [x] Part 2 - Fundamentals
- [ ] Part 3 - Advanced Concepts
- [ ] Part 2 - Authentication
- [ ] Part 5 - Django ORM
- [ ] Part 6 - Class-Based Views
- [ ] Part 7 - Deployment

## Few comments

__Part 1 - Getting Started__

Started in 06/03/2019 - Enviroment, python, django, etc...
Finished in 06/03

Talked about the configuration system (I've adapted it to my self), creating the project and app (different comands), the initial folder and source structures. Something about the files and configuration. How to dealing with url (requests, respones), ...

Some commands ilustrated during this section

```bash
# to create virtual envoiroments with vitualenv (called venv)
$ virtual venv
$ venv\Scripts\activate

# to create a django project (called myproject)
$ django-admin startproject myproject

# to run the webproject
$ py manage.py runserver

# to create an app (called boards)
$ django-admin statapp boards

```

Some important snippets inside some files (just to remember)
```python
# in views.py
from django.http import HttpResponse

def home(request):
    return HttpResponse('Hello, World!')

# in urls.py (to include a view called home from the app named boards)
from boards import views

urlpatters = [
    url(r'^$', views.home, name='home'),
    url(r'^admin/', admin.site.urls),
]
```

__Part 2 - Fundamentals__

Started in 06/03/2019 - Models, Views, Templates, Testing, and Admin
Finished in 07/03

Realy a lot of new and exciting things :) started with project requiremets, uml diagrams, models, migration of database, the API to dealing with models. Unit tests (need to check it more). Using templates to improve the views, static files and also the well know Bootstrap. To finish the Django Admin app and how to use it.

__Part 3 - Advanced Concepts__

Started in 07/03/2019 - URLs and Forms

A realy import thing here is the importance given to the test development practice. Every peace of new code on views and url scripts the author make a correlated test. The URL patterns was also detailed during this third part of the tutorial.

It also talked about Reusable Templates

Example of some urlpatters and regex (much more one: [URL Patterns](https://simpleisbetterthancomplex.com/references/2016/10/10/url-patterns.html))


| |**Primary Key AutoField**
-----|-------
Regex | ```(?P<pk>\d+)```
Example | ```url(r'^questions/(?P<pk>\d+)/$', views.question, name='question')```
Valid URL | ```/questions/934/```
Captures | ```{'pk': '934'}```


| |**Slug Field**
-----|-------
Regex | ```(?P<slug>[-\w]+)```
Example | ```url(r'^posts/(?P<slug>[-\w]+)/$', views.post, name='post')```
Valid URL | ```/posts/hello-world/```
Captures | ```{'slug': 'hello-world'}```


| |**Slug Field with Primary Key**
-----|-------
Regex | ```(?P<slug>[-\w]+)-(?P<pk>\d+)```
Example | ```url(r'^blog/(?P<slug>[-\w]+)-(?P<pk>\d+)/$', views.blog_post, name='blog_post')```
Valid URL | ```/blog/hello-world-159/```
Captures | ```{'slug': 'hello-world', 'pk': '159'}```


| |**Django User Username**
-----|-------
Regex | ```(?P<username>[\w.@+-]+)```
Example | ```url(r'^profile/(?P<username>[\w.@+-]+)/$', views.user_profile, name='user_profile')```
Valid URL | ```/profile/vitorfs/```
Captures | ```{'username': 'vitorfs'}```


| |**Year**
-----|-------
Regex | ```(?P<year>[0-9]{4})```
Example | ```url(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive, name='year'```
Valid URL | ```/articles/2016/```
Captures | ```{'year': '2016'}```


| |**Year / Month**
-----|-------
Regex | ```(?P<year>[0-9]{4})/(?P<month>[0-9]{2})```
Example | ```url('r^articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/$', views.month_archive, name='month')```
Valid URL | ```/articles/2016/01```
Captures | ```{''year': '2016'; 'month': '01'}```




__Part 2 - Authentication__

__Part 5 - Django ORM__

__Part 6 - Class-Based Views__

__Part 7 - Deployment__


