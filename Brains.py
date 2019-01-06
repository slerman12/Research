import tensorflow as tf
import sonnet as snt
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


class MHDPA(snt.AbstractModule):
    def __init__(self, name="mhdpa"):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          entities: Entities tensor to perform relational attention on.
        Returns:
          relations: Relations tensor.
        """
        super(MHDPA, self).__init__(name=name)

        self._entities = None
        self._relations = None
        self._mlp_relations = None
        self._aggregate_relations = None
        self._most_salient_relations = None
        self._entity_mask = None
        self._original_entity_mask = None
        self._most_salient_entity_mask = None

    def _build(self, entities, key_size, value_size, num_heads, entity_mask=None):
        self._original_entities = self._entities = entities
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._entity_mask = self._original_entity_mask = entity_mask

        qkv_size = 2 * key_size + value_size
        total_size = qkv_size * num_heads  # Denote as F.
        qkv = snt.BatchApply(snt.Linear(total_size))(entities)
        qkv = snt.BatchApply(snt.LayerNorm())(qkv)

        # Denoted as N.
        num_entities = entities.get_shape().as_list()[1]

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = snt.BatchReshape([num_entities, num_heads, qkv_size])(qkv)

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

        # Normalize - even out entity distribution (sqrt due to softmax exponential)
        q *= key_size ** -0.5

        # [B, H, N, N]  TODO: relu?
        weights = tf.matmul(q, k, transpose_b=True)

        # TODO: gating with softmax? Or relu & divide by abs sum? Or shift by min negative? Or re-implement softmax?
        # TODO: or exploit softmax temp
        if entity_mask is None:
            weights = tf.nn.softmax(weights)
        else:
            assert entity_mask.shape[1] == num_entities
            weights, squared_entity_mask = self.masked_softmask(weights, entity_mask, True)

        # For distributional, normalizing to probas  TODO: maybe use logits instead?
        self._context_saliences = snt.BatchFlatten()(tf.reduce_sum(weights, axis=2))
        self._context_saliences /= tf.tile(tf.reduce_sum(self._context_saliences, 1)[:, tf.newaxis], [1, num_entities])

        # [B, H, N, V]
        output = tf.matmul(weights, v)

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        # [B, N, H, V] -> [B, N, H * V]
        self._original_relations = self._relations = snt.BatchFlatten(preserve_dims=2)(output_transpose)

        return self._relations

    def apply_mlp_to_relations(self, output_size=None, residual_type=None):
        if output_size is None:
            output_size = [self._key_size, self._key_size]
        relations = snt.BatchApply(snt.nets.mlp.MLP(output_sizes=output_size))(self._relations)
        relations = snt.BatchApply(snt.LayerNorm())(relations)  # Normalization

        # Residual - enables contextualization, since entities generalizable, and these "relations" - contexts - merely
        # move it in a (layer norm!) direction. Layer norm does not appear to get applied again, because magnitude is
        # now meaningful: entity space modified by a direction; magnitude almost allows reconstruction of entity, since
        # entity itself has a layer norm applied.

        # Layer norm is 0 mean and unit variance - does it suffice, or would it be better to constrain divergence
        # rather than force it?

        # They even call them saliences.

        # Why do they use multiple heads? Just to account for their low order relations (entities), I suppose.

        # Oh! NOTE: they apply layer norm AFTER residual. But they do this everywhere, even relational RNN.

        # In any case, these skip connections probably suffice to improve generalization. Wouldn't a layer norm after
        # the residual hurt generalization?

        # Also note; "multiple blocks iwth shared (recurrent) or unshared (deep) parameters can be composed to
        # approximate higher order relations" - oy vey! That's my temporal and disentangled plan! Maybe study
        # "message passing on graphs"

        # Their baseline control agent replaces the relational blocks with "residual convolution" blocks!
        # Doesn't work as well (actually, seems okay!). not sure if they mean my simplified Contextual Relations method.
        # Mine uses full context to weigh relations without pairing; I don't know how theirs aggregates a relation, or
        # if they just average them without context. MHDPS, which is pairwise, confers entities with context. So does
        # my simplified and full model.

        # They mention that one block (order) preformed best on low order relation task and multiple blocks for higher
        # order, furthering my insight that these should be disentangled to get best of both.

        # Shit - for certain action outputs: "args are produced from the output of the aggregation function, whereas
        # Args_x,y result from upsampling the output of teh relational module" upsampling; that's my distributional
        # method! And for visualizing attention weights, yep, they looked at rows. They even use the word "options."
        # Yikes, all the relations attend to agent's location (early ego).

        # Yeah, they explicitly mention hierarchical RL. (This is all in reference to Relational Deep RL paper./
        if residual_type is not None:
            if residual_type == "add":
                relations += self._entities
            elif residual_type == "concat":
                relations = tf.concat([relations, self._entities])

        # Apply mask
        if self._entity_mask is not None:
            mask = self._entity_mask if self._most_salient_entity_mask is None else self._most_salient_entity_mask
            entity_mask = tf.cast(tf.tile(tf.expand_dims(mask, 2), [1, 1, relations.shape[2]]), tf.float32)
            relations *= entity_mask

        # MLP relations
        self._relations = self._mlp_relations = relations

        return self._mlp_relations

    def aggregate_relations(self, method="max"):
        # Aggregate relations  TODO: mask! -infinity for max, nrormalize for mean
        if method == "max":
            self._relations = self._aggregate_relations = tf.math.reduce_max(self._relations, axis=1)
        elif method == "mean":
            self._relations = self._aggregate_relations = tf.math.reduce_mean(self._relations, axis=1)
        elif method == "concat":
            self._relations = self._aggregate_relations = snt.BatchFlatten()(self._relations)

        return self._aggregate_relations

    def keep_most_salient(self, top_k, sample=True, uniform_sample=False):
        self._ensure_is_connected()
        # Saliences
        saliences = self._context_saliences

        # Salience sampling
        if sample or uniform_sample:
            # Assuming context saliences are properly masked and a probability distribution
            z = -tf.log(-tf.log(tf.random_uniform(tf.shape(saliences), 0, 1)))
            if self._entity_mask is not None:
                # Mask out noise on non-existent entities
                z *= tf.cast(self._entity_mask, tf.float32)
            saliences = tf.log(saliences) + z if not uniform_sample else z  # "Uniform" just as a baseline comparison

        # Indices of most salient
        top_k_indices = tf.nn.top_k(saliences, k=top_k, sorted=True).indices
        batch_size = tf.shape(top_k_indices)[0]
        batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], (1, top_k_indices.shape[1]))
        top_k_indices = tf.stack([batch_indices, top_k_indices], axis=-1)

        # Retrieving most salient relations (entity-based contexts) here
        self._relations = self._most_salient_relations = tf.gather_nd(self._relations, top_k_indices)
        self._entities = self._most_salient_entities = tf.gather_nd(self._entities, top_k_indices)

        # Update entity mask
        if self._entity_mask is not None:
            self._entity_mask = self._most_salient_entity_mask = tf.gather_nd(self._entity_mask, top_k_indices)

        return self._most_salient_relations

    def masked_softmask(self, weights, mask, square=False):
        # If mask needs to be applied to rows and columns of a square
        if square:
            edge_length = mask.get_shape().as_list()[1]
            mask = tf.expand_dims(mask, axis=1)  # [B, 1, N]
            mask = tf.tile(mask, [1, weights.shape[1], 1])  # [B, H, N]
            mask = tf.expand_dims(mask, axis=3)  # [B, H, N, 1]
            mask = tf.tile(mask, [1, 1, 1, edge_length])  # [B, H, N, N]
            mask = tf.math.logical_and(mask, tf.transpose(mask, [0, 1, 3, 2]))

        # (Masked) sparse representation
        indices = tf.where(mask)
        weights = tf.gather_nd(weights, indices)
        dense_shape = tf.cast(tf.shape(mask), tf.int64)
        sparse_result = tf.sparse_softmax(tf.SparseTensor(indices, weights, dense_shape))  # Softmax

        # Dense reconstruction
        weights = tf.scatter_nd(sparse_result.indices, sparse_result.values, sparse_result.dense_shape)
        weights.set_shape(mask.shape)
        return weights, mask


class SampleWithoutReplacement(snt.AbstractModule):
    def __init__(self, name="sample_without_replacement"):
        super(SampleWithoutReplacement, self).__init__(name=name)

    def _build(self, data, logits, num_samples):
        # Distribution based on logits
        dist = tfd.Categorical(logits=logits)

        # Get sample indices
        sample_indices, _ = self._get_sample_indices(dist, 3)

        # Batch coordinates (num_samples x num_batches)
        range_indices = tf.tile(tf.range(tf.shape(logits)[0])[tf.newaxis, ...], [tf.shape(sample_indices)[0], 1])

        # Sample indices paired with batch coordinates of shape (num_samples x num_batches x 2)
        data_indices = tf.stack([range_indices, sample_indices], -1)

        # Sampled items (num_samples x num_batches x sample_size)
        sampled = tf.gather_nd(data, data_indices)

        # Set shape
        sampled.set_shape([num_samples, None, sampled.shape[2]])

        self.bla = tf.transpose(sampled, [1, 0, 2])

        # Return batch-major sampled (num_batches x num_samples x sample_size)
        return tf.transpose(sampled, [1, 0, 2])

    def _get_sample_indices(self, categorical, num_samples):
        # For each sample
        _, growing_list_of_sample_indices, logits = tf.while_loop(
            lambda n, *_: n < num_samples,
            # Sample one by one without replacement, with initial iterator 0, empty list of sample indices, and logits
            # Note that the sample list is sample-major, not batch-major
            self._while_loop_body, [0,
                                    tf.zeros(tf.concat([[0], categorical.batch_shape_tensor(),
                                                        categorical.event_shape_tensor()], 0), dtype=categorical.dtype),
                                    categorical.logits],
            # Sample list grows with each iteration
            shape_invariants=[tf.TensorShape([]), tf.TensorShape(None), categorical.logits.shape])

        # Return sample list and distribution
        return growing_list_of_sample_indices, tfd.Categorical(logits=logits)

    def _while_loop_body(self, n, growing_list_of_sample_indices, logits):
        # Distribution
        dist = tfd.Categorical(logits=logits)

        # Sample
        sample = dist.sample()

        # One hot vector for categorical index sampled
        logit_mask = tf.one_hot(sample, tf.shape(logits)[-1], True, False)

        # Whichever index sampled, fill that with -infinity
        new_logits = tf.where(logit_mask, tf.fill(tf.shape(logits), tf.cast(-np.Inf, logits.dtype)), logits)

        # Add sample index to running list of sample indices
        growing_list_of_sample_indices = tf.concat([growing_list_of_sample_indices, sample[tf.newaxis, ...]], 0)

        # Return updated while loop signature
        return n + 1, growing_list_of_sample_indices, new_logits
