from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    # ================================
    # Load Base Model
    # ================================
    def get_base_model(self):
        if not Path(self.config.updated_base_model_path).exists():
            raise FileNotFoundError(
                f"Base model not found at: {self.config.updated_base_model_path}"
            )

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        print(f"[INFO] Base model loaded from: {self.config.updated_base_model_path}")

    # ================================
    # Data Generators
    # ================================
    def train_valid_generator(self):

        # --------- Path validation ---------
        data_path = Path(self.config.training_data)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Training data directory not found: {data_path}"
            )

        if not any(data_path.iterdir()):
            raise ValueError(
                f"Training directory is empty: {data_path}"
            )

        print(f"[INFO] Using dataset path: {data_path}")

        # --------- Generator configs ---------
        datagen_kwargs = {
            "rescale": 1./255,
            "validation_split": 0.20
        }

        dataflow_kwargs = {
            "target_size": self.config.params_image_size[:-1],
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear",
            "class_mode": "categorical"
        }

        # --------- Validation Generator ---------
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=str(data_path),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # --------- Training Generator ---------
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagen_kwargs
            )
        else:
            train_datagen = valid_datagen

        self.train_generator = train_datagen.flow_from_directory(
            directory=str(data_path),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # --------- Logging ---------
        print("\n[INFO] Dataset Summary")
        print(f"Classes        : {self.train_generator.class_indices}")
        print(f"Train samples  : {self.train_generator.samples}")
        print(f"Val samples    : {self.valid_generator.samples}")
        print(f"Image size     : {self.config.params_image_size}")
        print(f"Batch size     : {self.config.params_batch_size}")
        print(f"Augmentation   : {self.config.params_is_augmentation}")
        print("-" * 50)

    # ================================
    # Training
    # ================================
    def train(self, callback_list: list):

        if self.model is None:
            raise RuntimeError("Model not loaded. Call get_base_model() first.")

        if self.train_generator is None or self.valid_generator is None:
            raise RuntimeError("Generators not initialized. Call train_valid_generator() first.")

        # --------- Steps ---------
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print(f"[INFO] Steps per epoch      : {self.steps_per_epoch}")
        print(f"[INFO] Validation steps     : {self.validation_steps}")
        print(f"[INFO] Epochs               : {self.config.params_epochs}")

        # ================================
        # 🔥 COMPILE MODEL (SAFE & CLEAN)
        # ================================
        lr = self.config.params_learning_rate

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("[INFO] Model compiled successfully")

        # ================================
        # Train
        # ================================
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=callback_list
        )

        # ================================
        # Save Model
        # ================================
        self.save_model(
            path=Path(self.config.trained_model_path),
            model=self.model
        )

        print(f"[INFO] Model saved at: {self.config.trained_model_path}")

        return history

    # ================================
    # Save Model
    # ================================
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
