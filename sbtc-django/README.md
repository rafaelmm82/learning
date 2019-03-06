---
Just a test README.md file
---

# SBTC - A Complete Beginner's Guide to Django

This repo is to tracking the progress on my efforts to learning something about Django Framework. I'll starting following the Tutorials Series given by Vitor Freitas on his site [SimpleIsBetterThanComplex](http://simpleisbetterthancomplex.com/).

## Table of contents

The Tutorials Seires has sevens (7) parts, named:

- [x] Part 1 - Getting Started
- [ ] Part 2 - Fundamentals
- [ ] Part 3 - Advanced Concepts
- [ ] Part 2 - Authentication
- [ ] Part 5 - Django ORM
- [ ] Part 6 - Class-Based Views
- [ ] Part 7 - Deployment

## Few comments

__Part 1 - Getting Started__

06/03/2019 - Enviroment, python, django, etc...

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

06/03/2019 - Models, Views, Templates, Testing, and Admin



__Part 3 - Advanced Concepts__

__Part 2 - Authentication__

__Part 5 - Django ORM__

__Part 6 - Class-Based Views__

__Part 7 - Deployment__


