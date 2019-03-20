
# SBTC - A Complete Beginner's Guide to Django

This repo is to tracking the progress on my efforts to learning something about Django Framework. I'll starting following the Tutorials Series given by Vitor Freitas on his site [SimpleIsBetterThanComplex](http://simpleisbetterthancomplex.com/).

## Table of contents

The Tutorials Seires has sevens (7) parts, named:

- [x] Part 1 - Getting Started
- [x] Part 2 - Fundamentals
- [x] Part 3 - Advanced Concepts
- [x] Part 4 - Authentication
- [x] Part 5 - Django ORM
- [x] Part 6 - Class-Based Views
- [x] Part 7 - Deployment

## Few comments

__Part 1 - Getting Started__

<Started on 06/03/2019> - Enviroment, python, django, etc... 
<Finished on 06/03>

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

<Started on 06/03/2019> - Models, Views, Templates, Testing, and Admin 
<Finished on 07/03>

Realy a lot of new and exciting things :) started with project requiremets, uml diagrams, models, migration of database, the API to dealing with models. Unit tests (need to check it more). Using templates to improve the views, static files and also the well know Bootstrap. To finish the Django Admin app and how to use it.

__Part 3 - Advanced Concepts__

<Started on 07/03/2019> - URLs and Forms 
<Finished on 11/03>

A really import thing here is the importance given to the test development practice. Every peace of new code on views and url scripts the author make a correlated test. The URL patterns was also detailed during this third part of the tutorial.

It also talked about Reusable Templates. I continuing in the tutorial with the forms content. The csrf_token to dealing with forms is interesting also. The wrong (manual way) and te right way (builtin app) was clarifing. The test procedures became bigger but also important.

The Forms API changed all the way and makes a lot of hard work with validation, interaction and communicatino job much more efficient and secure.

Important command to install django-widget-tweaks

```
pip install django-widget-tweaks
```
and add in settings app section
```
INSTALLED_APPS = [
    'django.contrib.admin',
    ...
    'django.contrib.staticfiles',

    'widget_tweaks',

    'boards',
]
```

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




__Part 4 - Authentication__
<Started on 13/03> - Sign Up and dealing with users, password, reset, etc... 
<Finished on 14/03>

This part four was intense about tests, actually doubled the total number of tests. The login and password reset thing could be done almost by the auth API, especially the views. It is good.

It's totally ok to think on the accounts as an app, there are a lot of small, detailed, and important things related to it, in a unique way. The project became fragmented and complicated. It's a challenge to maintain without any documentation and business rules.


__Part 5 - Django ORM__
<Started on 15/03> - Protecting views, topics listing, Django ORM, migrations...
<Finished on 15/03>

As usual, the same as the last days, this part was really intense too. Too many concepts were presented about the framework and new functionalities. To protect the views was used a decorator (I need to learn more about this concept).

A lot of increment in the templates, views and also a little one in the models were done. The QuerySets was presented as a quite useful tool for dealing with objects in the DB.


__Part 6 - Class-Based Views__
<Started on 18/03> - Class Based Views, Pagination, Markdown ...
<Finished on 19/03>

About the Views strategies is important to reinforce the three types of views: Function-Based Views (FBV), Class-Based Views (CBV), and Generic Class-Based Views (GCBV). Each one with it's own importance and applications.

During this part we could make some content update, listing the vies with a lot of contents like the topics view and posts view. Markdown editing was also amazing, humanizing and Gravatar conclude the basics stuff dealing with people and web systems. There is a lot of but for the basics I think that everything is realy good.


__Part 7 - Deployment__
<Started on 19/03> - Deployment
<Finished on 19/03>


The last part was just about how to deploy the entire app over a server and connect it with a real DB and email server, get things secure with certifications, create a flow of source control, testing and put in production. It wi'll help us in our project but in a further moment.

Really great one tutorial that could be renamed as a Course. Thanks a lot to the Author.

