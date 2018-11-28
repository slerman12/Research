import tensorflow as tf
import sonnet as snt


class MHDPA(snt.AbstractModule):
    def __init__(self, name="mhdpa"):
        super(MHDPA, self).__init__(name=name)

    def _build(self, entities, key_size, value_size, num_heads, entity_mask=None):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          entities: Entities tensor to perform relational attention on.
        Returns:
          relations: Relations tensor.
        """
        self._entities = entities

        qkv_size = 2 * key_size + value_size
        total_size = qkv_size * num_heads  # Denote as F.
        qkv = snt.BatchApply(snt.Linear(total_size))(entities)
        qkv = snt.BatchApply(snt.LayerNorm())(qkv)

        num_entities = entities.get_shape().as_list()[1]  # Denoted as N.

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = snt.BatchReshape([num_entities, num_heads, qkv_size])(qkv)

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

        q *= key_size ** -0.5
        weights = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]

        if entity_mask is not None:
            assert entity_mask.get_shape().as_list()[1] == num_entities
            entity_mask = tf.expand_dims(entity_mask, axis=1)  # [B, 1, N]
            entity_mask = tf.tile(entity_mask, [1, weights.shape[1], 1])  # [B, H, N]
            entity_mask = tf.expand_dims(entity_mask, axis=3)  # [B, H, N, 1]
            entity_mask = tf.tile(entity_mask, [1, 1, 1, num_entities])  # [B, H, N, N]
            tf.math.logical_and(entity_mask, tf.transpose(entity_mask, [0, 1, 3, 2]))

            indices = tf.where(entity_mask)
            values = tf.gather_nd(weights, indices)
            dense_shape = tf.cast(tf.shape(weights), tf.int64)

            sparse_result = tf.sparse_softmax(tf.SparseTensor(indices, values, dense_shape))
            weights = tf.scatter_nd(sparse_result.indices, sparse_result.values, sparse_result.dense_shape)
            weights.set_shape(entity_mask.shape)
        else:
            weights = tf.nn.softmax(weights)

        self._entity_weights = tf.reduce_sum(weights, axis=2)  # For distributional relational reasoning

        output = tf.matmul(weights, v)  # [B, H, N, V]

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        # [B, N, H, V] -> [B, N, H * V]
        self._relations = snt.BatchFlatten(preserve_dims=2)(output_transpose)

        return self._relations

    def distributional(self, top_k, sample=True):
        self._ensure_is_connected()
        if top_k > 0:
            if sample:
                pass
            else:
                _, top_k_indices = tf.math.top_k(self._entity_weights, k=top_k)
                return tf.gather_nd(self._relations, top_k_indices)

