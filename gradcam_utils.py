import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm

def get_gradcam_heatmap(model, image_array, class_index, last_conv_layer_name="Conv_1"):
    """
    Generate Grad-CAM heatmap for a given image and class index.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image_array, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_gradcam_on_image(image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    Returns a NumPy array of the superimposed image.
    """
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # remove alpha channel
    heatmap_colored = np.uint8(255 * heatmap_colored)

    superimposed_img = heatmap_colored * alpha + image
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img 