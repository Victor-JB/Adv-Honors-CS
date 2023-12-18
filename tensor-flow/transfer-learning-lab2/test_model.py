
def main():
    if CHECKPOINT_EVAL_PATH:

        try:
            os.path.isdir(CHECKPOINT_EVAL_PATH)
        except:
            print(f"\nPath {CHECKPOINT_EVAL_PATH} is not a valid checkpoint path.")

        print(f"\nCheckpoint at {CHECKPOINT_EVAL_PATH} exist... loading in model from most recent checkpoint")

        if len(os.listdir(CHECKPOINT_EVAL_PATH)) > 1:
            most_recent = max([int(dir.split('_')[1]) for dir in os.listdir(CHECKPOINT_EVAL_PATH)])
            CHECKPOINT_EVAL_PATH = f'{CHECKPOINT_EVAL_PATH}/checkpoints_{most_recent}'

        model = tf.keras.models.load_model(ckpt_path)

        print(f"\nModel at '{ckpt_path}' loaded successfully; model structure: ")
        print(model.summary())

        print("\nEvaluating it with loaded dataset...")

        loss, acc = model.evaluate(ds_test, verbose=1)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))