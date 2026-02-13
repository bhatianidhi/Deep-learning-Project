import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.valid_generator = None
        self.score = None

    # ================================
    # Validation Generator
    # ================================
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            shuffle=False
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # DEBUG LOGS (IMPORTANT)
        print("\n[INFO] Evaluation Dataset Path:", self.config.training_data)
        print("[INFO] Absolute Path:", self.config.training_data.resolve())
        print("[INFO] Exists:", self.config.training_data.exists())
        print("-" * 50)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),   # ✅ ONLY CONFIG PATH
            **dataflow_kwargs
        )

    # ================================
    # Load Model
    # ================================
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    # ================================
    # Evaluation
    # ================================
    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

        print("\n[INFO] Evaluation completed")
        print(f"[INFO] Loss     : {self.score[0]}")
        print(f"[INFO] Accuracy : {self.score[1]}")

    # ================================
    # Save Scores
    # ================================
    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }
        save_json(path=Path("scores.json"), data=scores)
        print("[INFO] Scores saved to scores.json")
