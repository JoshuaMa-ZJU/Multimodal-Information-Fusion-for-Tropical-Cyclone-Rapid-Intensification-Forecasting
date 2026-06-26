import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_circular_masks(size=256, center=None):
    """Create six radial masks for storm-centered cloud features."""
    if center is None:
        center = (size // 2, size // 2)
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)

    masks = np.zeros((size, size, 6), dtype=np.float32)
    bins = [20, 40, 60, 80, 100, 120]
    masks[:, :, 0] = radius < bins[0]
    for idx in range(1, len(bins)):
        masks[:, :, idx] = (radius >= bins[idx - 1]) & (radius < bins[idx])
    return masks


def create_directional_masks(size=128, center=None):
    """Create eight directional masks for GPH-based storm-structure features."""
    if center is None:
        center = (size // 2, size // 2)
    yy, xx = np.ogrid[:size, :size]
    angles = np.degrees(np.arctan2(xx - center[1], yy - center[0]))

    masks = np.zeros((size, size, 8), dtype=np.float32)
    edges = [-135, -90, -45, 0, 45, 90, 135]
    masks[:, :, 0] = angles < edges[0]
    for idx in range(1, len(edges)):
        masks[:, :, idx] = (angles >= edges[idx - 1]) & (angles < edges[idx])
    masks[:, :, 7] = angles >= edges[-1]
    return masks


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.embed_dim, 2) * (-np.log(10000.0) / self.embed_dim)
        )
        pe = np.zeros((self.max_len, self.embed_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=self.dtype)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "embed_dim": self.embed_dim})
        return config


class DPTransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dp_dense = layers.Dense(embed_dim)
        self.fusion_dense = layers.Dense(embed_dim)
        self.output_dense = layers.Dense(embed_dim)

    def call(self, inputs, dp_features, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        dp_features = self.dp_dense(dp_features)
        dp_features = tf.expand_dims(dp_features, axis=1)
        dp_features = tf.repeat(dp_features, repeats=tf.shape(out1)[1], axis=1)
        fused = layers.add([dp_features, out1])
        fused = self.fusion_dense(fused)
        fused = layers.concatenate([fused, out1])
        out1 = self.output_dense(fused)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


def _expand_mask(mask):
    return tf.constant(mask[np.newaxis, np.newaxis, ...], dtype=tf.float32)


def build_model(
    cmask=None,
    dmask=None,
    history_steps=4,
    forecast_steps=4,
    embed_dim=256,
    num_heads=8,
    ff_dim=512,
):
    """Build the diurnal-pulse-aware multimodal forecasting model."""
    cmask = create_circular_masks() if cmask is None else cmask.astype(np.float32)
    dmask = create_directional_masks() if dmask is None else dmask.astype(np.float32)

    gridded_inputs = keras.Input(shape=(history_steps, 256, 256, 9), name="x")
    tabular_inputs = keras.Input(shape=(history_steps, 6), name="x_2d")
    dp_inputs = keras.Input(shape=(256, 256), name="dp")
    decoder_inputs = keras.Input(shape=(forecast_steps, 1), name="decoder_inputs")

    dp_features = layers.Reshape((256, 256, 1))(dp_inputs)
    dp_features = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(dp_features)
    dp_features = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(dp_features)
    dp_features = layers.GlobalMaxPooling2D()(dp_features)

    img = layers.Lambda(lambda x: x[:, :, :, :, 0:1])(gridded_inputs)
    gph = layers.Lambda(lambda x: x[:, :, :, :, 1:7])(gridded_inputs)
    sst = layers.Lambda(lambda x: x[:, :, :, :, 7:8])(gridded_inputs)
    sss = layers.Lambda(lambda x: x[:, :, :, :, 8:9])(gridded_inputs)

    img = layers.TimeDistributed(
        layers.Conv2D(1, 3, strides=1, padding="same", use_bias=False)
    )(img)
    img = layers.Multiply()([img, _expand_mask(cmask)])
    img_weights = layers.TimeDistributed(layers.GlobalAveragePooling2D())(img)
    img_weights = layers.TimeDistributed(layers.Dense(6, activation="sigmoid"))(img_weights)
    img_weights = layers.Reshape((history_steps, 1, 1, 6))(img_weights)
    img = layers.Multiply()([img_weights, img])
    img = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(img)
    img = layers.TimeDistributed(
        layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)
    )(img)

    sss_q = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sss)
    sss_k = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sss)
    sss_v = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sss)
    sst_q = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sst)
    sst_k = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sst)
    sst_v = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False))(sst)

    sst_att = layers.Multiply()([sss_q, sst_k])
    sst_att = layers.TimeDistributed(
        layers.Conv2D(32, 3, padding="same", activation="softmax", use_bias=False)
    )(sst_att)
    sst_att = layers.Multiply()([sst_att, sst_v])

    sss_att = layers.Multiply()([sst_q, sss_k])
    sss_att = layers.TimeDistributed(
        layers.Conv2D(32, 3, padding="same", activation="softmax", use_bias=False)
    )(sss_att)
    sss_att = layers.Multiply()([sss_att, sss_v])

    ocean = layers.Concatenate(axis=-1)([sss_att, sst_att])
    ocean = layers.TimeDistributed(
        layers.Conv2D(64, 3, strides=2, padding="same", activation="softmax", use_bias=False)
    )(ocean)

    gph = layers.TimeDistributed(layers.Conv2D(64, 3, strides=2, padding="same"))(gph)
    gph = layers.Reshape((history_steps, 128, 128, 64, 1))(gph)
    gph = layers.TimeDistributed(layers.Conv3D(1, 3, padding="same"))(gph)
    gph = layers.Reshape((history_steps, 128, 128, 64))(gph)
    gph = layers.TimeDistributed(layers.Conv2D(1, 1, padding="same"))(gph)
    gph_original = gph
    gph_masked = layers.Multiply()([gph, _expand_mask(dmask)])
    gph_weights = layers.TimeDistributed(layers.GlobalAveragePooling2D())(gph_masked)
    gph_weights = layers.TimeDistributed(layers.Dense(8, activation="softmax"))(gph_weights)
    gph_weights = layers.Reshape((history_steps, 1, 1, 8))(gph_weights)
    gph_masked = layers.Multiply()([gph_weights, gph_masked])
    gph_masked = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(gph_masked)
    gph = layers.Concatenate(axis=-1)([gph_original, gph_masked])
    gph = layers.TimeDistributed(layers.Conv2D(64, 1, strides=2, padding="same"))(gph)

    img = layers.TimeDistributed(layers.GlobalMaxPooling2D())(img)
    gph = layers.TimeDistributed(layers.GlobalMaxPooling2D())(gph)
    ocean = layers.TimeDistributed(layers.GlobalMaxPooling2D())(ocean)
    tabular = layers.TimeDistributed(layers.Dense(32, use_bias=False))(tabular_inputs)
    tabular = layers.TimeDistributed(layers.Dense(64, use_bias=False))(tabular)

    encoder_inputs = layers.Concatenate(axis=-1)([img, gph, ocean, tabular])
    pos_encoder = PositionalEncoding(history_steps, embed_dim)
    encoder_outputs = pos_encoder(encoder_inputs)
    encoder_outputs = DPTransformerEncoder(embed_dim, num_heads, ff_dim)(
        encoder_outputs, dp_features
    )

    decoder_outputs = layers.Dense(embed_dim)(decoder_inputs)
    decoder_outputs = PositionalEncoding(forecast_steps, embed_dim)(decoder_outputs)
    self_attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        decoder_outputs, decoder_outputs
    )
    decoder_outputs = layers.LayerNormalization()(decoder_outputs + self_attn_output)
    cross_attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        decoder_outputs, encoder_outputs
    )
    decoder_outputs = layers.LayerNormalization()(decoder_outputs + cross_attn_output)

    dp_decoder = layers.Dense(embed_dim)(dp_features)
    dp_decoder = layers.Reshape((1, embed_dim))(dp_decoder)
    dp_decoder = layers.Lambda(
        lambda x: tf.repeat(x, repeats=forecast_steps, axis=1),
        name="repeat_dp_decoder_features",
    )(dp_decoder)
    fused = layers.add([dp_decoder, decoder_outputs])
    fused = layers.Dense(embed_dim)(fused)
    decoder_outputs = layers.Dense(embed_dim)(layers.concatenate([fused, decoder_outputs]))

    ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
    decoder_outputs = layers.LayerNormalization()(decoder_outputs + ffn(decoder_outputs))
    outputs = layers.Dense(1, activation="linear")(decoder_outputs)
    outputs = layers.Lambda(lambda x: tf.cast(x, tf.float32), name="output_float32")(outputs)

    return keras.Model(
        [gridded_inputs, tabular_inputs, dp_inputs, decoder_inputs],
        outputs,
        name="diurnal_pulse_aware_tc_intensity_model",
    )
