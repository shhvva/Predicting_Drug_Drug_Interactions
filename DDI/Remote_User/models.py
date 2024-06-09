from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)

class drug_drug_interactions(models.Model):

    uniqueID=models.CharField(max_length=300)
    First_drugName=models.CharField(max_length=300)
    condition1=models.CharField(max_length=300)
    review=models.CharField(max_length=30000)
    rating=models.CharField(max_length=300)
    Second_drugName=models.CharField(max_length=300)
    condition2=models.CharField(max_length=300)
    Prediction=models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


