
import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint_path", required = True,
               help = "Checkpoint path from which to load model")
ap.add_argument("-i", "--test_image_path", required = False,
               help = "Image path from which to test with one image")
ap.add_argument("-d", "--test_dataset_path", required = False,
               help = "Dataset path with test images")
ap.add_argument("-e", "--eval_ds_path", required = False,
               help = "Path with dataset from which to evaluate the model")
args = vars(ap.parse_args())

def main():

    model = tf.keras.models.load_model(args['checkpoint_path'])
    print(f"\nModel at '{args['checkpoint_path']}' loaded successfully\n")
    model.summary()

    if args['test_image_path']:
        

    if args['eval_ds_path']:
        print(f"\n\nEvaluating it with dataset at {args['eval_ds_path']}")

        ds_train, ds_test, NUM_CLASSES = load_dataset(args['eval_ds_path'])

        # import load_dataset from training file...
        loss, acc = model.evaluate(ds_test, verbose=1)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
