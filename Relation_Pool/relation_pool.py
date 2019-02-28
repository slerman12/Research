import tensorflow as tf
import sonnet as snt


class RelationPool(snt.AbstractModule):
    def __init__(self, entities, k, name="relation_pool"):
        """Initialize pool with passed in entities
        Args:
            entities: original entities to be related
            k: size of relation pool to be maintained
        """
        super(RelationPool, self).__init__(name=name)

        # Size of each entity/relation
        self._entity_size = entities.get_shape().as_list()[-1]

        # Original entities, represented
        self._entities = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 4))(entities)

        # Pool level size
        self._k = k

        # Discover contexts for entities  TODO represent entities as relations first
        contexts, weights = self._discover_relational_context(entities)

        # Initiate pool with entities k most salient and set highest level
        indices = self._salience_sampling(weights, self._k)
        self._relations = tf.gather_nd(entities, indices)
        self._contexts = tf.gather_nd(contexts, indices)
        self._highest_level_relations = tf.gather_nd(self._entities, indices)

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

        # Return pool
        return self._relations, self._contexts

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

    def _add_to_pool(self, relations, contexts):
        """Adds relations and relational contexts to pool
        Args:
            relations: relations to add
            contexts: relational contexts to add
        """
        # Concatenate relations and contexts to existing pool
        self._relations = tf.concat([self._relations, relations], axis=-2)
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
            qkv_size = 2 * key_size + value_size
            qkv = snt.BatchApply(snt.Linear(qkv_size))(relations)
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
            if context_source is None:
                contexts += relations
            else:
                contexts += context_source

        # Returns relational contexts
        return contexts, weights

    def _salience_sampling(self, weights, k, deterministic=False, uniform_sample=False):
        """Discovering a relational context
        Args:
            weights: weights for each relation to each relation
            k: number of relations to select and reduce to
            deterministic: whether to sample deterministically
            uniform_sample: whether to sample uniformly
        Returns:
            indices for k most salient
        """
        self.weights = weights
        self.bla = snt.BatchFlatten()(tf.reduce_sum(weights, axis=1))
        self.blabla = snt.BatchFlatten()(tf.reduce_sum(weights, axis=2))
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

    def _confidence_sampling(self, relations, contexts):
        pass

    def infer(self, desired_outputs, output_shape):
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




