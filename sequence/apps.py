from django.apps import AppConfig
from .knn_model_for_protein_scaffold.knn_model import loaded_model


class SequenceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sequence'
    def ready(self):
        # Initialize the variables when the app is ready
        self.knn_classifier, self.CHAR_TO_INT, self.INT_TO_CHAR = loaded_model()
    def get_knn_model(self):
        # This method allows you to access the model attributes
        return self.knn_classifier, self.CHAR_TO_INT, self.INT_TO_CHAR
