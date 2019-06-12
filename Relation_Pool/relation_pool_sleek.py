import tensorflow as tf
import sonnet as snt


class RelationPool(snt.AbstractModule):
    def __init__(self, k, level, error_func, context_aggregation="max_pool", mode="salience",
                 aggregate_preds=False, generate_experts=False, name="relation_pool"):
        """Initialize pool with passed in entities
        Args:
            entities: original entities to be related
            k: size of relation pool to be maintained
        """
        super(RelationPool, self).__init__(name=name)
        # Pool level size
        self._k = k

        # Levels
        self._level = level

        # Compute error
        self._error_func = error_func

        # Aggregation type
        self._aggregation = context_aggregation

        # Pool mode
        self._mode = mode

        # Whether to aggregate predictions
        self._aggregate_preds = aggregate_preds

        # Whether to generate experts
        self._generate_experts = generate_experts

        # Expert generator
        if self._generate_experts:
            self._expert_generator = snt.nets.mlp.MLP([64, 64 * 64])

        # Loss
        self.loss = tf.constant(0, dtype=tf.float32)

    def _build(self, entities):
        """Consider each entity and relation pair-wise, add these to pool, reduce pool to size k
        Args:
            entities: entities to perform relational computations on
        Returns:
            relation: a final extracted relation
        """
        # Size of each entity/relation
        self._entity_size = entities.get_shape().as_list()[-1]

        # Original entities, represented
        self._entities = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size] * 4))(entities)

        # Discover contexts for entities
        contexts, weights = self._discover_relational_context(self._entities)

        # Global context
        if self._aggregation == "lstm":
            # For lstm, add contexts to loss, then pass ordered entities to self._aggregate
            self.add_contexts_to_loss(contexts)
            indices, _ = self._salience_sampling(weights, deterministic=True)
            self._global_context = self._aggregate(tf.reverse(tf.gather_nd(self._entities, indices), [1]), mode="lstm")
        elif self._aggregation == "basic":
            self._global_context = self._basic_relation_representation(self._entities)
        elif self._aggregation == "mlp":
            self._global_context = snt.nets.mlp.MLP([256, 256, 256, 256])(snt.BatchFlatten()(self._entities))
        else:
            self._global_context = self._aggregate(contexts, mode=self._aggregation)

        # If generating experts, generate context query from global context
        if self._generate_experts:
            self._context_query = snt.Linear(self._entity_size)(self._global_context)

        # Global query for valuing confidence of relations
        self._context_query_for_valuing_relations = snt.Linear(self._entity_size)(self._global_context)
        self._context_query_for_valuing_relations = snt.LayerNorm()(self._context_query_for_valuing_relations)
        self._context_query_for_valuing_relations *= self._entity_size ** -0.5
        # Global key for valuing confidence of relations
        self._key_generator_for_valuing_relations = snt.BatchApply(snt.Linear(self._entity_size))

        # Get indices of entities k most salient
        if self._mode == "salience":
            indices, _ = self._salience_sampling(weights, self._k)
            self._contexts = tf.gather_nd(contexts, indices)
        elif self._mode == "confidence":
            indices, _ = self._confidence_sampling(self._entities, self._global_context, self._k)

        # Relations and highest level
        self._relations = self._highest_level_relations = tf.gather_nd(self._entities, indices)

        # Iterate up the relation hierarchy
        for _ in range(self._level - 1):
            if self._mode == "salience":
                self._iterate_via_salience_sampling()
            elif self._mode == "confidence":
                self._iterate_via_confidence_sampling()

        # Return final relation
        if self._mode == "salience":
            return self._output_via_salience_sampling()
        elif self._mode == "confidence":
            return self._output_via_confidence_sampling()

    def _iterate_via_confidence_sampling(self):
        """Consider each entity and relation pair-wise, add these to pool, reduce pool to size k
        """
        # Get relations and contexts from pairing each relation in pool and entity
        relations_pairwise = self._pairwise_with_entities(self._highest_level_relations)

        # Represent relations
        relations_represented = self._represent_relations_of_entities(relations_pairwise)

        # Return top k indices
        indices, _ = self._confidence_sampling(relations_represented, self._global_context, self._k)

        # Top k relations
        relations = tf.gather_nd(relations_represented, indices)

        # # Their corresponding confidences
        # confidences = tf.gather_nd(confidences, indices)
        #
        # # Add to loss
        # self._add_confidences_to_loss(confidences, relations)

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
        relations_represented = self._represent_relations_of_entities(relations_pairwise)

        # Discover contexts (Note: self attention is performed level-wise, not on entities as the context source)
        contexts, weights = self._discover_relational_context(relations_represented)

        # Return top k indices
        indices, _ = self._salience_sampling(weights, self._k)

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

    def _basic_relation_representation(self, relations, size=None):
        """Represent entities as a basic relation representation
        Args:
            relations
        Returns:
            relation_representations: each relation represented
        """
        # Run an MLP over each entity in relation
        mlp_entities = snt.BatchApply(snt.nets.mlp.MLP([self._entity_size]*4
                                                       if size is None else [size]*2))(relations)

        # Aggregate the MLP'ed entities
        entity_aggregation = tf.reduce_sum(mlp_entities, axis=1)

        # MLP the aggregation to get final representation of relation
        relation_representations = snt.nets.mlp.MLP([self._entity_size] * 2
                                                    if size is None else [size] * 2)(entity_aggregation)

        # Return representations
        return relation_representations

        # return self._self_attention(relations)

    def _represent_relations_of_entities(self, relations):
        """Represent concatenated entities as a relation representation
        Args:
            relations
        Returns:
            Each relation represented
        """
        # Return representations
        return snt.BatchApply(self._basic_relation_representation)(relations)
        # return snt.BatchApply(self._self_attention)(relations)

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
            lstm = tf.nn.rnn_cell.LSTMCell(self._entity_size, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
            _, state = tf.nn.dynamic_rnn(lstm, to_aggregate, dtype=tf.float32)
            aggregate = state.h
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

    def _discover_relational_context(self, attending, attended=None, residual=True):
        """Discovering a relational context
        Args:
            attending
            residual: whether to use a residual
        Returns:
            contexts: relational contexts for relations in pool
            weights: weights for each relation to each relation
        """
        # Key and value size (default: relation size)
        key_size = self._entity_size
        value_size = self._entity_size

        # Separate attending relations & attended context source s.t. entities may constitute the context
        if attended is None:
            # Linearly project queries, keys, and values from relations
            qkv = snt.BatchApply(snt.Linear(2 * key_size + value_size))(attending)
            qkv = snt.BatchApply(snt.LayerNorm())(qkv)

            # Queries, keys, and values
            queries, keys, values = tf.split(qkv, [key_size, key_size, value_size], -1)
        else:
            # Linearly project queries
            queries = snt.BatchApply(snt.Linear(key_size))(attending)
            queries = snt.BatchApply(snt.LayerNorm())(queries)

            # Linearly project keys and values
            kv = snt.BatchApply(snt.Linear(key_size + value_size))(attended)
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
            contexts += attending

        # Returns relational contexts
        return contexts, weights

    def _attend_one_to_many(self, one, many):
        """Attending a generated query across a set of generated keys
        Args:
            one: query source
            many: source of keys
        Returns:
            weights: weights for each item of many
        """
        # Linearly project query [B x E]
        # query = snt.nets.mlp.MLP([self._entity_size])(one)
        # query = snt.LayerNorm()(query)

        # Normalize (even out relation distribution; sqrt due to softmax exponential)
        # query *= self._entity_size ** -0.5

        # Linearly project keys [B x N x E]
        keys = self._key_generator_for_valuing_relations(many)
        keys = snt.BatchApply(snt.LayerNorm())(keys)

        # Weights [B x N x 1]
        # weights = tf.matmul(keys, query[:, tf.newaxis, :], transpose_b=True)
        weights = tf.matmul(keys, self._context_query_for_valuing_relations[:, tf.newaxis, :], transpose_b=True)

        # [B x N]
        return tf.squeeze(weights, axis=2)

    def _self_attention(self, attending):
        """Self attention
        Args:
            attending: items to do self-attention on
        Returns:
            attended
        """
        contexts, _ = self._discover_relational_context(attending)
        return self._aggregate(contexts)

    def _salience_sampling(self, weights, k=None, deterministic=False, uniform_sample=False):
        """Indices based on 'salience sampling'
        Args:
            weights: weights for each relation to each relation
            k: number of relations to select and reduce to
            deterministic: whether to sample deterministically
            uniform_sample: whether to sample uniformly
        Returns:
            indices for k most salient
        """
        # Default k to all
        if k is None:
            k = weights.get_shape().as_list()[2]

        # Saliences
        saliences = tf.reduce_sum(weights, axis=1)
        saliences /= tf.tile(tf.reduce_sum(saliences, axis=1)[:, tf.newaxis], [1, weights.shape[2]])

        # Sampling
        return self.sample(saliences, k, deterministic, uniform_sample), saliences

    def _confidence_sampling(self, relations, context, k, deterministic=False, uniform_sample=False, add_to_loss=False):
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
        # Attend context to relations
        weights = self._attend_one_to_many(context, relations)
        # contexts, weights = self._discover_relational_context(context[:, tf.newaxis, :], relations, residual=False)

        # Add contexts to loss
        # self.add_contexts_to_loss(contexts)

        # Confidence probas
        confidences = tf.nn.softmax(weights, axis=1)
        # confidences = tf.squeeze(weights, axis=1)

        # Add to loss
        if add_to_loss:
            if self._aggregate_preds:
                self._add_aggregate_loss(confidences, relations)
            else:
                self._add_confidences_to_loss(confidences, relations)

            if self._generate_experts:
                self._add_experts_to_loss(confidences, relations, inference_func=True)

        # Sampling
        return self.sample(confidences, k, deterministic, uniform_sample), confidences

    def sample(self, probas, k, deterministic, uniform_sample):
        # Sampling
        if not deterministic:
            z = -tf.log(-tf.log(tf.random_uniform(tf.shape(probas), 0, 1)))
            probas = tf.log(probas) + z if not uniform_sample else z

        # Indices of most salient
        top_k_indices = tf.nn.top_k(probas, k=k, sorted=True).indices
        batch_size = tf.shape(top_k_indices)[0]
        batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], (1, top_k_indices.shape[1]))
        top_k_indices = tf.stack([batch_indices, top_k_indices], axis=-1)

        # Indices of most confidence and confidences
        return top_k_indices

    def _add_aggregate_loss(self, confidences, relations):
        """Confidence sampling loss (aggregate predictions)
        Args:
            confidences, relations
        """
        errors = self._error_func(relations, confidences=confidences)
        self.loss += tf.reduce_mean(errors)

    def _add_confidences_to_loss(self, confidences, relations):
        """Confidence sampling loss
        Args:
            confidences, relations
        """
        errors = self._error_func(relations)
        self.loss += tf.reduce_mean(errors * confidences)

    def _add_experts_to_loss(self, confidences, relations, inference_func=None):
        """Confidence sampling loss
        Args:
            confidences, relations
        """
        experts = snt.BatchApply(self._expert_generator)(relations)
        experts = tf.reshape(experts, shape=[-1, experts.shape[1], 64, 64])
        component_knowledge = tf.einsum('ijkl,il->ijk', experts, self._context_query)

        errors = self._error_func(component_knowledge, inference_func=inference_func)
        self.loss += tf.reduce_mean(errors * confidences)

    def add_relations_to_loss(self, relations):
        """Salience sampling loss
        Args:
            relations
        """
        # Loss
        errors = self._error_func(relations)
        self.loss += tf.reduce_mean(errors)

    def add_contexts_to_loss(self, contexts):
        """Salience sampling loss
        Args:
            contexts
        """
        # Loss
        errors = self._error_func(contexts, mode="contexts")
        self.loss += tf.reduce_mean(errors)

    def _output_via_confidence_sampling(self):
        """Output using relation pool
        Returns:
            final_relation: model output
        """
        # Index of predictor and predicted errors
        indices, confidences = self._confidence_sampling(self._relations, self._global_context, self._k,
                                                         add_to_loss=True)

        # Most confident relations
        final_relations = tf.gather_nd(self._relations, indices)
        final_confidences = tf.gather_nd(confidences, indices)
        final_confidences /= tf.tile(tf.reduce_sum(final_confidences, axis=1)[:, tf.newaxis],
                                     [1, final_confidences.shape[1]])

        # # Most confident relation
        # batch_range = tf.range(tf.shape(final_relations)[0])
        # first_index_of_each_batch = tf.stack([batch_range, tf.zeros(tf.shape(final_relations)[0], tf.int32)], axis=1)
        # final_relation = tf.gather_nd(final_relations, first_index_of_each_batch)

        # Return prediction and loss
        return final_relations, final_confidences

    def _output_via_salience_sampling(self):
        """Output using relation pool
        Returns:
            final_relation: model output
        """
        # Most salient indices for relations in pool
        contexts, weights = self._discover_relational_context(self._relations)
        indices, saliences = self._salience_sampling(weights, self._k)

        # Most salient relations
        final_relations = tf.gather_nd(self._relations, indices)
        final_saliences = tf.gather_nd(saliences, indices)
        final_saliences /= tf.tile(tf.reduce_sum(final_saliences, axis=1)[:, tf.newaxis],
                                   [1, final_saliences.shape[1]])

        # Their associated contexts
        final_contexts = tf.gather_nd(contexts, indices)

        # All contexts [B x Level*K+K x Entity Size]
        all_contexts = tf.concat([self._contexts, final_contexts], axis=1)

        # Loss
        self.add_relations_to_loss(final_relations)
        self.add_contexts_to_loss(all_contexts)

        # Most salient relation
        # batch_range = tf.range(tf.shape(final_relations)[0])
        # first_index_of_each_batch = tf.stack([batch_range, tf.zeros(tf.shape(final_relations)[0], tf.int32)], axis=1)
        # final_relation = tf.gather_nd(final_relations, first_index_of_each_batch)

        # Return prediction and loss
        return final_relations, final_saliences
    
    def aggregate_outputs(self, relations, weights, inference_func):
        """Output prediction via aggregation
        Args:
            relations, weights, function for inference
        Returns:
            aggregated relations according to weights
        """
        # Inferences
        inferences = tf.nn.softmax(snt.BatchApply(inference_func)(relations))
        
        # Weights
        weights = tf.tile(weights[:, :, tf.newaxis], [1, 1, inferences.shspe[2]])
        
        # Return aggregation
        return tf.reduce_sum(inferences * weights, axis=1)

    def first_relation_output(self, relations, inference_func):
        """Output prediction
        Args:
            relations, function for inference
        Returns:
            output of first relation
        """
        # Highest weight relation
        batch_range = tf.range(tf.shape(relations)[0])
        first_index_of_each_batch = tf.stack([batch_range, tf.zeros(tf.shape(relations)[0], tf.int32)], axis=1)
        relation = tf.gather_nd(relations, first_index_of_each_batch)
    
        # Return prediction
        return inference_func(relation)
