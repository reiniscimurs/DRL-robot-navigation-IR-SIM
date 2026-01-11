import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def entropy_from_attention(attn_weights, target_nodes, num_nodes, eps=1e-10):
    """
    Compute mean per-node entropy of incoming softmax attention distributions.

    Args:
        attn_weights (Tensor): Per-edge attention probabilities of shape (num_edges,).
        target_nodes (Tensor): Destination node indices for each edge of shape (num_edges,).
        num_nodes (int): Total number of nodes/agents.
        eps (float, optional): Small constant for numerical stability in log. Defaults to 1e-10.

    Returns:
        Tensor: Scalar tensor containing the mean entropy across nodes that have at least
            one incoming edge. Returns 0.0 if no nodes have incoming edges.
    """

    # Compute log(p), safe for zero
    attn_log = (attn_weights + eps).log()

    # For each edge, p*log(p)
    contrib = -(attn_weights * attn_log)  # [num_edges]

    # Sum contributions per node (i.e., sum over incoming edges)
    entropies = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, contrib
    )
    counts = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, torch.ones_like(attn_weights)
    )

    # (Optional) Only average over nodes with at least one incoming edge
    mask = counts > 0
    entropies = entropies[mask] / counts[mask]
    return (
        entropies.mean()
        if mask.any()
        else torch.tensor(0.0, device=attn_weights.device)
    )


class GoalAttentionLayer(MessagePassing):
    """
    Message-passing layer with learned attention over goal/edge features.

    This layer computes attention scores from concatenated node queries and
    transformed edge attributes, then aggregates value projections with 'add'
    aggregation.

    Args:
        node_dim (int): Dimensionality of node features.
        edge_dim (int): Dimensionality of edge attributes.
        out_dim (int): Output dimensionality for projected messages.
    """

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr="add")  # Could use 'mean' or 'max' as well.
        self.q = torch.nn.Linear(node_dim, out_dim, bias=False)
        self.k = torch.nn.Linear(edge_dim, out_dim, bias=False)
        self.v = torch.nn.Linear(edge_dim, out_dim)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )
        self._last_attn_weights = None

    def forward(self, x, edge_index, edge_attr):
        """
        Run attention-based message passing.

        Args:
            x (Tensor): Node features of shape (num_nodes, node_dim).
            edge_index (LongTensor): Edge indices of shape (2, num_edges)
                with format [source, target].
            edge_attr (Tensor): Edge attributes of shape (num_edges, edge_dim).

        Returns:
            tuple:
                out (Tensor): Updated node features of shape (num_nodes, out_dim).
                attn_weights (Tensor or None): Last per-edge softmax weights of
                    shape (num_edges,). None if no edges.
        """
        q = self.q(x)
        out = self.propagate(edge_index, x=q, edge_attr=edge_attr)
        return out, self._last_attn_weights

    def message(self, x_i, edge_attr, index, ptr, size_i):
        """
        Compute per-edge messages using attention weights.

        Args:
            x_i (Tensor): Target-node queries for each edge, shape (num_edges, out_dim).
            edge_attr (Tensor): Edge attributes for each edge, shape (num_edges, edge_dim).
            index (LongTensor): Target node indices per edge, shape (num_edges,).
            ptr: Unused (PyG internal).
            size_i: Unused (PyG internal).

        Returns:
            Tensor: Per-edge messages of shape (num_edges, out_dim).
        """
        k = F.leaky_relu(self.k(edge_attr))
        v = F.leaky_relu(self.v(edge_attr))
        attention_input = torch.cat([x_i, k], dim=-1)
        scores = self.attn_score_layer(attention_input).squeeze(-1)
        attn_weights = softmax(scores, index)  # index = target node index
        self._last_attn_weights = attn_weights.detach()
        return v * attn_weights.unsqueeze(-1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate messages per target node.

        Args:
            inputs (Tensor): Per-edge messages, shape (num_edges, out_dim).
            index (LongTensor): Target node indices per edge, shape (num_edges,).
            ptr: Unused (PyG internal).
            dim_size (int or None): Number of target nodes (optional).

        Returns:
            Tensor: Aggregated node features, shape (num_nodes, out_dim).
        """
        return super().aggregate(inputs, index, ptr, dim_size)


class Attention(nn.Module):
    """
    Multi-robot attention mechanism combining hard (binary) and soft (weighted) attentions.

    The module encodes agent features, derives pairwise geometric edge attributes,
    computes hard attention masks and soft attention weights, performs message passing,
    and decodes concatenated self/attended embeddings.

    Args:
        embedding_dim (int): Dimension of the agent embedding vector.

    Attributes:
        embedding1 (nn.Linear): First layer for agent feature encoding.
        embedding2 (nn.Linear): Second layer for agent feature encoding.
        hard_mlp (nn.Sequential): MLP over concatenated node/edge features for hard attention.
        hard_encoding (nn.Linear): Outputs logits for binary attention (2 classes).
        q (nn.Linear): Query projection for soft attention.
        k (nn.Linear): Key projection from edge features for soft attention.
        v (nn.Linear): Value projection from edge features for soft attention.
        attn_score_layer (nn.Sequential): Computes unnormalized scores for soft attention.
        decode_1 (nn.Linear): First decoder layer for concatenated embeddings.
        decode_2 (nn.Linear): Second decoder layer for concatenated embeddings.
        message_graph (GoalAttentionLayer): Graph attention/message-passing layer.
        node_update (nn.Sequential): Optional node-update MLP (unused in forward aggregation).
    """

    def __init__(self, embedding_dim):
        """
        Initialize the attention module.

        Args:
            embedding_dim (int): Output embedding dimension per agent.
        """
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim

        self.node_update = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

        self.message_graph = GoalAttentionLayer(
            node_dim=embedding_dim, edge_dim=10, out_dim=embedding_dim
        )

        self.embedding1 = nn.Linear(5, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")

        # Hard attention MLP with distance
        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Soft attention projections
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(10, embedding_dim, bias=False)
        self.v = nn.Linear(10, embedding_dim)

        # Soft attention score network (with polar other robot goal position)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # Decoder
        self.decode_1 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_1.weight, nonlinearity="leaky_relu")
        self.decode_2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_2.weight, nonlinearity="leaky_relu")

    def encode_agent_features(self, embed):
        """
        Encode per-agent features with a two-layer MLP.

        Args:
            embed (Tensor): Raw agent features of shape (B*N, 5).

        Returns:
            Tensor: Encoded embeddings of shape (B*N, embedding_dim).
        """
        embed = F.leaky_relu(self.embedding1(embed))
        embed = F.leaky_relu(self.embedding2(embed))
        return embed

    def forward(self, embedding):
        """
        Compute hard and soft attentions and produce attended embeddings.

        Args:
            embedding (Tensor): Input tensor of shape (B, N, D), where D ≥ 11. The first
                channels are expected to include position (x,y), heading (cos,sin),
                agent features for encoding, action, and goal.

        Returns:
            tuple:
                att_embedding (Tensor): Attended embedding, shape (B*N, 2*embedding_dim).
                hard_logits (Tensor): Logits for hard attention (keep class index 1),
                    shape (B*N, N-1).
                unnorm_rel_dist (Tensor): Unnormalized pairwise distances, shape (B*N, N-1, 1).
                mean_entropy (Tensor): Scalar mean entropy of soft attention per batch.
                hard_weights (Tensor): Binary hard attention mask, shape (B, N, N).
                comb_w (Tensor): Combined soft weights per (receiver, sender),
                    shape (N, N*(N-1)).
        """
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        batch_size, n_agents, _ = embedding.shape

        # Extract sub-features
        embed = embedding[:, :, 4:9].reshape(batch_size * n_agents, -1)
        position = embedding[:, :, :2].reshape(batch_size, n_agents, 2)
        heading = embedding[:, :, 2:4].reshape(
            batch_size, n_agents, 2
        )  # assume (cos(θ), sin(θ))
        action = embedding[:, :, 7:9].reshape(batch_size, n_agents, 2)
        goal = embedding[:, :, -2:].reshape(batch_size, n_agents, 2)

        # Compute pairwise relative goal vectors (for each i,j)
        goal_j = goal.unsqueeze(1).expand(-1, n_agents, -1, -1)
        pos_i = position.unsqueeze(2)
        goal_rel_vec = goal_j - pos_i

        # Encode agent features
        agent_embed = self.encode_agent_features(embed)
        agent_embed = agent_embed.view(batch_size, n_agents, self.embedding_dim)

        # Prep for hard attention: compute all relative geometry for each agent pair
        h_i = agent_embed.unsqueeze(2)  # (B, N, 1, D)
        pos_i = position.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = position.unsqueeze(1)  # (B, 1, N, 2)
        heading_i = heading.unsqueeze(2)  # (B, N, 1, 2)
        heading_j = heading.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (B, 1, N, 2)

        # Compute relative vectors and distance
        rel_vec = pos_j - pos_i  # (B, N, N, 2)
        dx, dy = rel_vec[..., 0], rel_vec[..., 1]
        rel_dist = (
            torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True) / 12
        )  # (B, N, N, 1)

        # Relative angle in agent i's frame
        angle = torch.atan2(dy, dx) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_sin = torch.sin(angle)
        rel_angle_cos = torch.cos(angle)

        # Other agent's heading
        heading_j_cos = heading_j[..., 0]  # (B, 1, N)
        heading_j_sin = heading_j[..., 1]  # (B, 1, N)

        # Edge features for hard attention
        edge_features = torch.cat(
            [
                rel_dist,  # (B, N, N, 1)
                rel_angle_cos.unsqueeze(-1),  # (B, N, N, 1)
                rel_angle_sin.unsqueeze(-1),  # (B, N, N, 1)
                heading_j_cos.unsqueeze(-1),  # (B, N, N, 1)
                heading_j_sin.unsqueeze(-1),  # (B, N, N, 1)
                action.unsqueeze(1).expand(-1, n_agents, -1, -1),  # (B, N, N, 2)
            ],
            dim=-1,
        )

        # Broadcast agent embedding for all pairs (except self-pairs)
        h_i_expanded = h_i.expand(-1, -1, n_agents, -1)

        h_i_flat = h_i_expanded.reshape(
            batch_size * n_agents, n_agents, self.embedding_dim
        )
        edge_flat = edge_features.reshape(batch_size * n_agents, n_agents, -1)

        # Concatenate agent embedding and edge features
        hard_input = torch.cat([h_i_flat, edge_flat], dim=-1)

        # Hard attention forward
        h_hard = self.hard_mlp(hard_input)
        hard_logits = self.hard_encoding(h_hard)
        hard_weights = F.gumbel_softmax(hard_logits, hard=False, tau=0.2, dim=-1)[
            ..., 1
        ].unsqueeze(2)
        hard_weights = hard_weights.view(batch_size, n_agents, n_agents)

        unnorm_rel_dist = torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True)
        unnorm_rel_dist = unnorm_rel_dist.reshape(batch_size * n_agents, n_agents, 1)

        # ---- Soft attention computation ----
        q = self.q(agent_embed).reshape(batch_size * n_agents, -1)

        attention_outputs = []
        entropy_list = []
        combined_w = []

        # Goal-relative polar features for soft attention
        goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
        goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
        heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
        goal_rel_angle = goal_angle_global - heading_angle
        goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
        goal_rel_angle_cos = torch.cos(goal_rel_angle).unsqueeze(-1)
        goal_rel_angle_sin = torch.sin(goal_rel_angle).unsqueeze(-1)
        goal_polar = torch.cat(
            [goal_rel_dist, goal_rel_angle_cos, goal_rel_angle_sin], dim=-1
        )

        # Soft attention edge features (include goal polar)
        soft_edge_features = torch.cat([edge_features, goal_polar], dim=-1)

        attn_outputs = []

        for b in range(batch_size):
            edge_index_list = []
            edge_attr_list = []

            # Agent embeddings for this scenario
            node_feats = agent_embed[b]  # [n, emb_dim]
            soft_feats = soft_edge_features[b]  # [n, n, edge_dim]
            hard_mask = hard_weights[b]  # [n, n]

            # Only build edges where hard attention mask is active, no self-loops
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j and hard_mask[i, j] > 0.5:
                        edge_index_list.append([j, i])  # from j (src) to i (tgt)
                        edge_attr_list.append(soft_feats[i, j])

            # Edge tensors
            if edge_index_list:
                edge_index = torch.tensor(
                    edge_index_list, dtype=torch.long, device=embedding.device
                ).t()  # [2, num_edges]
                edge_attr = torch.stack(edge_attr_list, dim=0)  # [num_edges, edge_dim]
            else:
                edge_index = torch.zeros(
                    (2, 0), dtype=torch.long, device=embedding.device
                )
                edge_attr = torch.zeros(
                    (0, soft_feats.shape[-1]),
                    dtype=soft_feats.dtype,
                    device=embedding.device,
                )

            # Message passing: edge-to-node aggregation only
            attn_out, attn_weights = self.message_graph(
                node_feats, edge_index, edge_attr
            )  # [n, embedding_dim]
            attn_outputs.append(attn_out)
            batch_entropy = entropy_from_attention(
                attn_weights, edge_index[1], num_nodes=n_agents
            )
            entropy_list.append(batch_entropy)

            combined_weights = torch.zeros(
                (n_agents, n_agents), device=attn_weights.device
            )
            for idx in range(edge_index.shape[1]):
                j = edge_index[0, idx].item()  # sender/source
                i = edge_index[1, idx].item()  # receiver/target
                combined_weights[i, j] = attn_weights[idx]
            combined_w.append(combined_weights)

        # Re-stack results for full batch (shape: [batch_size, n_agents, embedding_dim])
        attn_stack = torch.stack(attn_outputs, dim=0).reshape(batch_size * n_agents, -1)
        self_embed = agent_embed.reshape(batch_size * n_agents, -1)

        # Concat original + attended for each agent
        concat_embed = torch.cat(
            [self_embed, attn_stack], dim=-1
        )  # [batch, n, 2*embedding_dim]

        x = F.leaky_relu(self.decode_1(concat_embed))
        att_embedding = F.leaky_relu(self.decode_2(x))
        mean_entropy = torch.stack(entropy_list).mean()
        comb_w = torch.stack(combined_w, dim=1).reshape(n_agents, -1)

        return (
            att_embedding,
            hard_logits[..., 1],
            unnorm_rel_dist,
            mean_entropy,
            hard_weights,
            comb_w,
        )
