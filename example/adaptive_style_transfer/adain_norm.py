import tensorflow as tf


# Normalizes the `content_features` with scaling and offset from `style_features`.
def AdaIN(content_features, style_features, alpha=1, epsilon=1e-5):

    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keep_dims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keep_dims=True)

    normalized_content_features = tf.nn.batch_normalization(
        content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon
    )
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features
