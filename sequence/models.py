from django.db import models

# Create your models here.
""" # Create the dropdown and text scaffolding models
class DropdownScaffolding(models.Model):
    dropdown_field = models.CharField(max_length=200)

class TxtScaffolding(models.Model):
    text_field = models.TextField()

# Save the dropdown and text scaffolding models
dropdown_obj = DropdownScaffolding.objects.create(dropdown_field=DropdownScaffolding.objects)
dropdown_obj.save()
text_obj = TxtScaffolding.objects.create(text_field='Hello!')
text_obj.save() """