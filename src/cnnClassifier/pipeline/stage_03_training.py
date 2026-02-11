from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training     
from cnnClassifier import logger
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()




STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
      config = ConfigurationManager()
      prepare_callbacks_config = config.get_prepare_callback_config()
      prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
      callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

      training_config = config.get_training_config()

    # 🔍 DEBUG HERE
      from pathlib import Path
      print("\n========== DEBUG ==========")
      print("training_data value:", training_config.training_data)
      print("type:", type(training_config.training_data))

      p = Path(training_config.training_data)
      print("Absolute path:", p.resolve())
      print("Exists:", p.exists())
      if p.exists():
         print("Contents:", list(p.iterdir()))
      print("===========================\n")

      training = Training(config=training_config)
      training.get_base_model()
      training.train_valid_generator()
      training.train(callback_list=callback_list)




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        

        