import tensorflow as tf
import sonnet as snt


class RelationPool(snt.AbstractModule):
    def __init__(self, entities, k, compute_error=None, output_shape=None, aggregation_mode="max_pool",
                 initiate_pool_mode="salience_sampling", name="relation_pool"):
        """Initialize pool with passed in entities
        Args:
            entities: original entities to be related
            k: size of relation pool to be maintained
        """
        super(RelationPool, self).__init__(name=name)

        # Size of each entity/relation
        self._entity_size = entities.get_shape().as_list()[-1]

        # Original entities, represented TODO should they be further represented like this?
        self._entities = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 4))(entities)

        # Pool level size
        self._k = k

        # Compute error of relations
        self._compute_error = compute_error

        # Loss
        self.loss = tf.constant(0, dtype=tf.float32)

        # Discover contexts for entities
        contexts, weights = self._discover_relational_context(entities)

        # Global context
        self._global_context = self._aggregate(contexts, mode=aggregation_mode)

        # Get indices of entities k most salient
        if initiate_pool_mode == "salience_sampling":
            indices = self._salience_sampling(weights, self._k)
        elif initiate_pool_mode == "confidence_sampling":
            self._inference = snt.nets.mlp.MLP([256, 256, 256, 256, output_shape])
            indices, confidences = self._confidence_sampling(self._entities, self._global_context, self._k)

        # Relations and highest level
        self._relations = self._highest_level_relations = tf.gather_nd(self._entities, indices)

        # Contexts for salience sampling
        self._contexts = tf.gather_nd(contexts, indices)

    def _build(self, level):
        """Consider each entity and relation pair-wise, add these to pool, reduce pool to size k
        Args:
            level: how many levels to iterate to, the maximum order of the relation hierarchy
        Returns:
            relations: a pool of relations
            contexts: their respective relational contexts
        """
        # Iterate up the relation hierarchy
        for _ in range(level if self.is_connected else level - 1):
            self._iterate_via_confidence_sampling()

        # Return pool
        return self._relations, self._contexts

    def _iterate_via_confidence_sampling(self):
        """Consider each entity and relation pair-wise, add these to pool, reduce pool to size k
        """
        # Get relations and contexts from pairing each relation in pool and entity
        relations_pairwise = self._pairwise_with_entities(self._highest_level_relations)

        # Represent relations
        relations_represented = self._relation_representation(relations_pairwise)

        # Return top k indices
        indices, confidences = self._confidence_sampling(relations_represented, self._global_context, self._k)

        # Top k relations
        relations = tf.gather_nd(relations_represented, indices)

        # Update highest level
        self._highest_level_relations = tf.gather_nd(relations_pairwise, indices)

        # Add these relations to pool
        self._add_to_pool(relations)

    def _iterate_via_salience_sampling(self):
        """Consider each entity and relation pair-wise, add these to pool, reduce pool to size k
        """
        # Get relations and contexts from pairing each relation in pool and entity
        relations_pairwise = self._pairwise_with_entities(self._highest_level_relations)

        # Represent relations
        relations_represented = self._relation_representation(relations_pairwise)

        # Discover contexts (Note: self attention is performed level-wise, not on entities as the context source)
        contexts, weights = self._discover_relational_context(relations_represented)

        # Return top k indices
        indices = self._salience_sampling(weights, self._k)

        # Top k relations and contexts
        relations = tf.gather_nd(relations_represented, indices)
        contexts = tf.gather_nd(contexts, indices)

        # Update highest level
        self._highest_level_relations = tf.gather_nd(relations_pairwise, indices)

        # Add these relations and contexts to pool
        self._add_to_pool(relations, contexts)

    def _pairwise_with_entities(self, relations):
        """Consider each entity and relation pair-wise
        Args:
            relations
        Returns:
            relations: each relation paired with each entity
        """
        # If relations constitute single entities, add an empty dimension
        if len(relations.shape) < 4:
            relations = relations[:, :, tf.newaxis, :]

        # Batch Size x Num Relations * Num Entities x Num Entities In Relation x Entity Size
        relations_squared = tf.tile(relations, [1, self._entities.shape[1], 1, 1])

        # Batch Size x Num Relations * Num Entities x Entity Size
        entities_squared = snt.BatchReshape([relations.get_shape().as_list()[1] *
                                             self._entities.get_shape().as_list()[1], self._entity_size])(
            tf.tile(self._entities, [1, 1, relations.shape[1]]))

        # Add axis and concatenate
        relations = tf.concat([relations_squared, entities_squared[:, :, tf.newaxis, :]], axis=2)

        return relations

    def _relation_representation(self, relations):
        """Represent concatenated entities as a relation representation
        Args:
            relations
        Returns:
            relation_representations: each relation represented
        """
        # Run an MLP over each entity in relation
        mlp_entities = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 4), n_dims=3)(relations)

        # Aggregate the MLP'ed entities
        entity_aggregation = tf.reduce_sum(mlp_entities, axis=2)

        # MLP the aggregation to get final representation of relation
        relation_representations = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 2))(entity_aggregation)

        # Return representations
        return relation_representations

    def _aggregate(self, to_aggregate, mode="max_pool"):
        """Aggregate inputs
        Args:
            to_aggregate: items to aggregate
            mode: method of aggregation
        Returns:
            aggregate: aggregated items
        """
        # Aggregate relations
        if mode == "max_pool":
            aggregate = tf.math.reduce_max(to_aggregate, axis=1)
        elif mode == "mean":
            aggregate = tf.math.reduce_mean(to_aggregate, axis=1)
        elif mode == "concat":
            aggregate = snt.BatchFlatten()(to_aggregate)
        elif mode == "lstm":
            pass
        return aggregate

    def _add_to_pool(self, relations, contexts=None):
        """Adds relations and relational contexts to pool
        Args:
            relations: relations to add
            contexts: relational contexts to add
        """
        # Concatenate relations and contexts to existing pool
        self._relations = tf.concat([self._relations, relations], axis=-2)
        if contexts is not None:
            self._contexts = tf.concat([self._contexts, contexts], axis=-2)

    def _discover_relational_context(self, relations, context_source=None, residual=True):
        """Discovering a relational context
        Args:
            relations
            residual: whether to use a residual
        Returns:
            contexts: relational contexts for relations in pool
            weights: weights for each relation to each relation
        """
        # Key and value size (default: relation size)
        key_size = self._entity_size
        value_size = self._entity_size

        # Separate attending relations & attended context source s.t. entities may constitute the context
        if context_source is None:
            # Linearly project queries, keys, and values from relations
            qkv = snt.BatchApply(snt.Linear(2 * key_size + value_size))(relations)
            qkv = snt.BatchApply(snt.LayerNorm())(qkv)

            # Queries, keys, and values
            queries, keys, values = tf.split(qkv, [key_size, key_size, value_size], -1)
        else:
            # Linearly project queries
            queries = snt.BatchApply(snt.Linear(key_size))(relations)
            queries = snt.BatchApply(snt.LayerNorm())(queries)

            # Linearly project keys and values
            kv = snt.BatchApply(snt.Linear(key_size + value_size))(context_source)
            kv = snt.BatchApply(snt.LayerNorm())(kv)

            # Keys and values
            keys, values = tf.split(kv, [key_size, value_size], -1)

        # Normalize (even out relation distribution; sqrt due to softmax exponential)
        queries *= key_size ** -0.5

        # Weights [B x N x N]
        weights = tf.nn.softmax(tf.matmul(queries, keys, transpose_b=True))

        # Relational contexts
        contexts = tf.matmul(weights, values)  # Context
        contexts = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 2))(contexts)  # MLP
        contexts = snt.BatchApply(snt.LayerNorm())(contexts)  # Layer normalization

        # Residual
        if residual:
            contexts += relations

        # Returns relational contexts
        return contexts, weights

    def _attend_one_to_many(self, one, many):
        # Linearly project query [B x E]
        query = snt.nets.mlp.MLP([self._entity_size] * 2)(one)
        query = snt.LayerNorm()(query)

        # Linearly project keys [B x N x E]
        keys = snt.BatchApply(snt.Linear(self._entity_size))(many)
        keys = snt.BatchApply(snt.LayerNorm())(keys)

        # Weights [B x N x 1]
        weights = tf.nn.softmax(tf.matmul(keys, query[:, tf.newaxis, :], transpose_b=True))
        print(weights)

        # [B x N]
        return tf.squeeze(weights, axis=2)

    def _salience_sampling(self, weights, k, deterministic=False, uniform_sample=False):
        """Indices based on 'salience sampling'
        Args:
            weights: weights for each relation to each relation
            k: number of relations to select and reduce to
            deterministic: whether to sample deterministically
            uniform_sample: whether to sample uniformly
        Returns:
            indices for k most salient
        """
        # Saliences
        saliences = tf.reduce_sum(weights, axis=1)
        saliences /= tf.tile(tf.reduce_sum(saliences, axis=1)[:, tf.newaxis], [1, weights.shape[2]])

        # Sampling
        if not deterministic:
            z = -tf.log(-tf.log(tf.random_uniform(tf.shape(saliences), 0, 1)))
            saliences = tf.log(saliences) + z if not uniform_sample else z

        # Indices of most salient
        top_k_indices = tf.nn.top_k(saliences, k=k, sorted=True).indices
        batch_size = tf.shape(top_k_indices)[0]
        batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], (1, top_k_indices.shape[1]))
        top_k_indices = tf.stack([batch_indices, top_k_indices], axis=-1)

        # Reducing pool to most salient relations
        return top_k_indices

    def _confidence_sampling(self, relations, context, k, deterministic=False, uniform_sample=False):
        """Indices based on 'confidence sampling'
        Args:
            relations, context
            k: number of relations to select and reduce to
            deterministic: whether to sample deterministically
            uniform_sample: whether to sample uniformly
        Returns:
            indices for k most confident
            predicted errors
        """
        # Concatenate relations with context
        # relation_paired_with_context = tf.concat([relations, tf.tile(context[:, tf.newaxis, :],
        #                                                              [1, relations.shape[1], 1])], axis=-1)

        # error_inference_source = tf.concat([tf.stop_gradient(relations),
        #                                     tf.tile(context[:, tf.newaxis, :], [1, relations.shape[1], 1])], axis=-1)

        # # Predicted errors of relations [B x N x 1]
        # predicted_error = snt.BatchApply(self._error_inference)(error_inference_source)
        #
        # # Remove exxtra dimension
        # predicted_error = tf.squeeze(predicted_error, axis=2)
        #
        # # Confidence
        # confidences = tf.nn.softmax(tf.divide(1, predicted_error), axis=1)

        # Confidence inference for confidence sampling
        # infer_confidence = snt.nets.mlp.MLP([256, 256, 256, 256, 1])

        # Predicted confidence of relations [B x N x 1] -> [B x N]
        # confidences = confidence_sampled = tf.nn.softmax(tf.squeeze(snt.BatchApply(infer_confidence)(
        #     relation_paired_with_context), axis=2), axis=1)

        # confidences = confidence_sampled = tf.nn.softmax(tf.squeeze(snt.BatchApply(self._infer_confidence)(
        #     tf.stop_gradient(relation_paired_with_context)), axis=2), axis=1)

        weights = self._attend_one_to_many(context, relations)
        confidences = tf.nn.softmax(weights, axis=1)
        predictions = snt.BatchApply(self._inference)(relations)

        self._add_confidences_to_loss(confidences, predictions)

        # Sampling
        if not deterministic:
            z = -tf.log(-tf.log(tf.random_uniform(tf.shape(confidences), 0, 1)))
            confidence_sampled = tf.log(confidences) + z if not uniform_sample else z

        # Indices of most salient
        top_k_indices = tf.nn.top_k(confidence_sampled, k=k, sorted=True).indices
        batch_size = tf.shape(top_k_indices)[0]
        batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], (1, top_k_indices.shape[1]))
        top_k_indices = tf.stack([batch_indices, top_k_indices], axis=-1)

        # Indices of most confidence and predicted errors NOTE: they do not align if sampling enabled!
        return top_k_indices, confidences

    def _add_confidences_to_loss(self, confidences, predictions):
        errors = self._compute_error(predictions)
        self.loss += tf.reduce_mean(errors * confidences)

    def _infer_via_confidence_sampling(self):
        """Inference using relation pool
        Args:
            compute_error: function to compute error between batch of relations' predictions and desired outputs
            desired_outputs: target labels
            output_shape: shape of output
        Returns:
            predictions: model outputs
            loss: all losses together
        """
        # Ensure built
        self._ensure_is_connected()

        # Predictions
        predictions = snt.BatchApply(self._inference)(self._relations)

        # Compute loss of each relation's prediction
        # errors = self._compute_error(predictions)

        # Index of predictor and predicted errors
        index, confidences = self._confidence_sampling(self._relations, self._global_context, 1)

        # indices, _ = self._confidence_sampling(self._relations, self._global_context, 10)
        # preds = snt.BatchApply(inference_pred)(tf.gather_nd(self._relations, indices))
        # final_errors = compute_error(preds, desired_outputs)

        self._add_confidences_to_loss(confidences, predictions)

        # Predictor
        final_relation = tf.gather_nd(self._relations, index)

        # Prediction
        prediction = self._inference(tf.squeeze(final_relation, axis=1))

        # Reshape predicted errors
        # predicted_errors = tf.reshape(predicted_errors, [-1])

        # context_prediction = snt.nets.mlp.MLP([256, 256, 256, 256, output_shape])(self._global_context)
        # context_loss = compute_error(context_prediction[:, tf.newaxis, :], desired_outputs)
        # context_loss = tf.reduce_mean(context_loss)

        # Loss
        # loss = tf.reduce_mean(errors) + tf.losses.mean_squared_error(errors, predicted_errors)
        # loss = tf.reduce_mean(final_errors) + tf.losses.mean_squared_error(errors, predicted_errors)
        # loss = tf.reduce_mean(final_errors) + tf.losses.mean_squared_error(tf.stop_gradient(errors), predicted_errors)
        # loss = tf.reduce_mean(errors) + tf.losses.mean_squared_error(tf.stop_gradient(errors), predicted_errors)
        # errors: [B x N], predicted_errors: [B x N]
        # loss = tf.reduce_mean(errors * confidences) + tf.reduce_mean(errors * tf.reshape(self._confidences,
        #                                                                                  [-1, errors.shape[1]]))
        # loss = tf.reduce_mean(errors * confidences) + context_loss

        # Return prediction and loss
        return prediction, self.loss

    def infer_via_salience_sampling(self, desired_outputs, output_shape):
        """Inference using relation pool
        Args:
            desired_outputs: target labels
            output_shape: shape of output
        Returns:
            predictions: model outputs
            loss: loss from desired outputs
        """
        # Ensure built
        self._ensure_is_connected()

        # Most salient indices for relations in pool
        pool_contexts, pool_weights = self._discover_relational_context(self._relations)
        pool_indices = self._salience_sampling(pool_weights, self._k)

        # Most salient relations and relation
        final_relations = tf.gather_nd(self._relations, pool_indices)
        batch_range = tf.range(tf.shape(final_relations)[0])
        first_index_of_each_batch = tf.stack([batch_range, tf.zeros(tf.shape(final_relations)[0], tf.int32)], axis=1)
        final_relation = tf.gather_nd(final_relations, first_index_of_each_batch)

        # Their associated contexts
        final_contexts = tf.gather_nd(pool_contexts, pool_indices)

        # All contexts [B x Level*K+K x Entity Size]
        all_contexts = tf.concat([self._contexts, final_contexts], axis=1)

        # Predictions
        # [B * Level*K+K x Output Shape]
        contextual_predictions = tf.reshape(snt.BatchApply(snt.nets.mlp.MLP([self._entity_size, output_shape]))(all_contexts), [-1, output_shape])
        relational_predictor = snt.nets.mlp.MLP([self._entity_size, self._entity_size, output_shape])
        # [B * K x Output Shape]
        relational_predictions = tf.reshape(snt.BatchApply(relational_predictor)(final_relations), [-1, output_shape])
        # [B x Output Shape]
        prediction = relational_predictor(final_relation)

        # Context loss
        # [B * Level*K+K]
        context_desired_outputs = tf.reshape(tf.tile(desired_outputs[:, tf.newaxis], [1, all_contexts.shape[1]]), [-1])
        context_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context_desired_outputs,
                                                                      logits=contextual_predictions)

        # Relation loss
        # [B * K]
        relation_desired_outputs = tf.reshape(tf.tile(desired_outputs[:, tf.newaxis], [1, final_relations.shape[1]]), [-1])
        relation_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=relation_desired_outputs,
                                                                       logits=relational_predictions)
        # Loss
        loss = tf.reduce_mean(context_loss) + tf.reduce_mean(relation_loss)

        # Return prediction and loss
        return prediction, loss




