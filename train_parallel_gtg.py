    # FILE: train_parallel_gtg.py
# DESCRIPTION: A new parallel GTG architecture inspired by GTC-Net,
#              where GNN and Transformer branches run in parallel and exchange information.

import os, pickle, math, time, warnings, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
# ... (所有 import 保持不变) ...
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import torch_geometric.nn as gnn
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.model_selection import KFold
from sklearn import metrics
from losses import WeightedCrossEntropy


# --- 1. 全局参数定义 (保持不变) ---
FEATURE_PATH = "./GraphPPIS_Feature/"
MODEL_PATH = "./models_saved_lu/"  # 新建目录
DATASET_PATH = "./GraphPPIS_Dataset/"
SEED = 2024;
np.random.seed(SEED);
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.set_device(0); torch.cuda.manual_seed(SEED)

IN_CHANNELS = 93; # pssm（20） hmm（20） dssp（14） resAF（7） esm32 93
MAP_CUTOFF = 8;  # 埃
EDGE_FEATURE_DIR = "edge_features_rbf16";
EDGE_DIM = 16  #rbf编码
HIDDEN_DIM = 128;   #64
NUM_GNN_LAYERS_PART1 = 1;#1
NUM_TRANS_LAYERS = 2;#2
NUM_GNN_LAYERS_PART2 = 3;#3
TRANS_HEADS = 8
DROPOUT = 0.5;# 0.5 0.1-0.5
NUM_CLASSES = 2;
ALPHA = 0.2;  #残差系数
ECA_K_SIZE = 7;
GAT_HEADS = 4
LEARNING_RATE = 5e-4;#5e-4
WEIGHT_DECAY = 1e-4;#1e-4
BATCH_SIZE = 1;
NUMBER_EPOCHS = 100;
PATIENCE = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. 模型架构定义 ---

# ... (PositionalEncoding 和所有 GNN 组件保持不变, 直接复制) ...
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__();
        self.dropout = nn.Dropout(p=dropout);
        pe = torch.zeros(max_len, d_model);
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1);
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model));
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term);
        pe = pe.unsqueeze(0);
        self.register_buffer('pe', pe)

    def forward(self, x): x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)


class GraphECALayer(nn.Module):
    def __init__(self, channel, k_size=ECA_K_SIZE): super().__init__(); self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                                                                                              padding=(k_size - 1) // 2,
                                                                                              bias=False); self.sigmoid = nn.Sigmoid()

    def forward(self, x): y = torch.mean(x, dim=0, keepdim=True); y = self.conv(y.unsqueeze(1)).squeeze(
        1); y = self.sigmoid(y); return x * y


class GCNIIStyleConv(nn.Module):
    def __init__(self, hidden_dim): super().__init__(); self.conv = gnn.GCNConv(hidden_dim,
                                                                                hidden_dim); self.weight1 = Parameter(
        torch.FloatTensor(hidden_dim, hidden_dim)); self.weight2 = Parameter(
        torch.FloatTensor(hidden_dim, hidden_dim)); self.reset_parameters()

    def reset_parameters(self): stdv = 1. / math.sqrt(self.weight1.size(1)); self.weight1.data.uniform_(-stdv,
                                                                                                        stdv); self.weight2.data.uniform_(
        -stdv, stdv)

    def forward(self, x, h0, edge_index, alpha): gcn_out = self.conv(x, edge_index); term1 = (1 - alpha) * torch.matmul(
        gcn_out, self.weight1); term2 = alpha * torch.matmul(h0, self.weight2); return term1 + term2


class EdgeAwareGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=GAT_HEADS, concat=False, dropout=0.0):
        super(EdgeAwareGATLayer, self).__init__(aggr='add', node_dim=0);
        self.in_channels, self.out_channels, self.edge_dim, self.heads, self.concat, self.dropout = in_channels, out_channels, edge_dim, heads, concat, dropout;
        self.out_dim_per_head = out_channels;
        self.lin = nn.Linear(in_channels, out_channels * heads);
        self.lin_edge = nn.Linear(edge_dim, out_channels * heads);
        self.att = nn.Parameter(torch.Tensor(1, heads, self.out_dim_per_head));
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight); glorot(self.lin_edge.weight); glorot(self.att)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x);
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        if self.concat:
            out = out.view(-1, self.heads * self.out_dim_per_head)
        else:
            out = out.mean(dim=1)
        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        x_i = x_i.view(-1, self.heads, self.out_dim_per_head);
        x_j = x_j.view(-1, self.heads, self.out_dim_per_head);
        edge_embedding = self.lin_edge(edge_attr).view(-1, self.heads, self.out_dim_per_head);
        alpha = F.leaky_relu((x_i + x_j) + edge_embedding);
        alpha = (alpha * self.att).sum(dim=-1);
        alpha = softmax(alpha, index, ptr, size_i);
        alpha = F.dropout(alpha, p=self.dropout, training=self.training);
        return x_j * alpha.unsqueeze(-1)


class HybridGNNBlock(nn.Module):
    def __init__(self, hidden_dim, alpha, dropout, edge_dim):
        super().__init__();
        self.gcn_layer = GCNIIStyleConv(hidden_dim);
        self.gat_layer = EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout);
        self.sage_layer = gnn.SAGEConv(hidden_dim, hidden_dim, aggr='mean');
        self.eca_gcn = GraphECALayer(hidden_dim);
        self.eca_gat = GraphECALayer(hidden_dim);
        self.eca_sage = GraphECALayer(hidden_dim);
        self.combiner = nn.Linear(hidden_dim * 3, hidden_dim);
        self.eca_final = GraphECALayer(hidden_dim);
        self.norm = nn.LayerNorm(hidden_dim);
        self.act_fn = nn.ReLU(inplace=True);
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, h0, edge_index, edge_attr):
        h_gcn = self.eca_gcn(self.gcn_layer(h, h0, edge_index, alpha=ALPHA));
        h_gat = self.eca_gat(self.gat_layer(h, edge_index, edge_attr=edge_attr));
        h_sage = self.eca_sage(self.sage_layer(h, edge_index));
        h_combined = torch.cat([h_gcn, h_gat, h_sage], dim=1);
        h_fused = self.combiner(h_combined);
        h_fused_calibrated = self.eca_final(h_fused);
        h_out = h + self.dropout(h_fused_calibrated);
        h_out = self.norm(h_out);
        h_out = self.act_fn(h_out);
        return h_out


# --- 【核心修改】: 新的 Parallel GTG Encoder with Gating ---
class Parallel_GTG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_proj = nn.Linear(IN_CHANNELS, HIDDEN_DIM)
        self.act_fn = nn.ReLU(inplace=True)

        # GNN Part 1 (初提)
        self.gnn_blocks_part1 = nn.ModuleList(
            [HybridGNNBlock(HIDDEN_DIM, ALPHA, DROPOUT, EDGE_DIM) for _ in range(NUM_GNN_LAYERS_PART1)])

        # 并行的 GNN Part 2 和 Transformer
        self.gnn_blocks_part2 = nn.ModuleList(
            [HybridGNNBlock(HIDDEN_DIM, ALPHA, DROPOUT, EDGE_DIM) for _ in range(NUM_GNN_LAYERS_PART2)])

        self.gat_layer = EdgeAwareGATLayer(HIDDEN_DIM, HIDDEN_DIM, edge_dim=EDGE_DIM, dropout=DROPOUT);

        self.pos_encoder = PositionalEncoding(HIDDEN_DIM, DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=TRANS_HEADS,
                                                   dim_feedforward=HIDDEN_DIM * 2, dropout=DROPOUT, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TRANS_LAYERS)

        # --- 从可学习标量 s1, s2 升级为门控单元 ---
        # 门控单元 1: 决定 GNN 输出应该接收多少 Transformer 信息
        self.gate_for_gnn = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.Sigmoid()
        )
        # 门控单元 2: 决定 Transformer 输出应该接收多少 GNN 信息
        self.gate_for_trans = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.Sigmoid()
        )

        self.norm_gnn_out = nn.LayerNorm(HIDDEN_DIM)
        self.norm_trans_out = nn.LayerNorm(HIDDEN_DIM)

        self.final_fusion = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.final_norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x, edge_index, edge_attr):
        # 1. 初始投影
        h = self.act_fn(self.initial_proj(x));
        h0 = h

        # 2. GNN Part 1: 特征初提
        h_gnn1 = h
        for block in self.gnn_blocks_part1:
            h_gnn1 = block(h_gnn1, h0, edge_index, edge_attr)

        # h_gnn1 = self.gat_layer(h_gnn1, edge_index, edge_attr)

        # 3. 并行计算
        # 3.1 GNN 分支
        h_gnn2_pure = h_gnn1
        # for block in self.gnn_blocks_part2:
        #     h_gnn2_pure = block(h_gnn2_pure, h0, edge_index, edge_attr)

        h_gnn2_pure = self.gat_layer(h_gnn2_pure, edge_index, edge_attr)

        # 3.2 Transformer 分支
        h_trans_in = h_gnn1.unsqueeze(0)
        h_with_pe = self.pos_encoder(h_trans_in)
        h_trans_pure = self.transformer(h_with_pe).squeeze(0)

        # 4. 【核心改进】双向门控信息注入
        # 4.1 GNN 接收 Transformer 信息
        gate_gnn_input = torch.cat([h_gnn2_pure, h_trans_pure], dim=1)
        g_gnn = self.gate_for_gnn(gate_gnn_input)
        h_gnn_fused = self.norm_gnn_out(h_gnn2_pure + g_gnn * h_trans_pure)

        # 4.2 Transformer 接收 GNN 信息
        gate_trans_input = torch.cat([h_trans_pure, h_gnn2_pure], dim=1)
        g_trans = self.gate_for_trans(gate_trans_input)
        h_trans_fused = self.norm_trans_out(h_trans_pure + g_trans * h_gnn2_pure)

        # 5. 最终融合 (保持不变)
        h_final = torch.cat([h_gnn_fused, h_trans_fused], dim=1)
        h_final = self.final_fusion(h_final)

        # 6. 全局残差连接 (保持不变)
        return self.final_norm(h_gnn1 + h_final)

class GraphPPIS_Parallel_GTG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Parallel_GTG_Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM // 2, NUM_CLASSES)
        )
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1]), reduction='sum')
        self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, edge_index, edge_attr):
        x = x.float().to(device);
        edge_index = edge_index.long().to(device);
        edge_attr = edge_attr.float().to(device)
        node_embeddings = self.encoder(x, edge_index, edge_attr)
        output = self.classifier(node_embeddings)
        return output


# --- 3 & 4. 数据处理、训练、评估逻辑 (完全不变, 直接复制) ---
# ... (ProDataset, load_edges, collate_fn, train_one_epoch, evaluate, analysis, train)
def load_edges(sequence_name):
    dismap_path = os.path.join(FEATURE_PATH, "distance_map", f"{sequence_name}.npy")
    dismap = np.load(dismap_path);
    mask = (dismap > 0) & (dismap <= MAP_CUTOFF);
    adj_for_edges = torch.tensor(mask, dtype=torch.float)
    return adj_for_edges.nonzero(as_tuple=False).t().contiguous()


class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values;
        self.sequences = dataframe['sequence'].values;
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        try:
            label = np.array(self.labels[index], dtype=np.int64)
            pssm = np.load(os.path.join(FEATURE_PATH, "pssm", f"{sequence_name}.npy"))
            hmm = np.load(os.path.join(FEATURE_PATH, "hmm", f"{sequence_name}.npy"))
            dssp = np.load(os.path.join(FEATURE_PATH, "dssp", f"{sequence_name}.npy"))
            af = np.load(os.path.join(FEATURE_PATH, "resAF", f"{sequence_name}.npy"))
            esm_ae_feature = np.load(os.path.join(FEATURE_PATH, "esm2_ae32", f"{sequence_name}.npy"))
            all_node_features_list = [pssm, hmm, dssp, af, esm_ae_feature]
            # all_node_features_list = [pssm, hmm, dssp, esm_ae_feature]
            valid_lengths = [f.shape[0] for f in all_node_features_list if f.ndim > 0 and f.shape[0] > 0]
            if not valid_lengths: return None
            min_len = min(valid_lengths)
            if len(label) < min_len: return None
            label = label[:min_len]
            aligned_features = [f[:min_len] for f in all_node_features_list]
            node_features = np.concatenate(aligned_features, axis=1).astype(np.float32)
            edge_index = load_edges(sequence_name);
            edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)
            if edge_index.numel() > 0:
                mask = (edge_index[0] < min_len) & (edge_index[1] < min_len)
                edge_index = edge_index[:, mask]
                if edge_index.numel() > 0:
                    try:
                        full_edge_features_raw = np.load(
                            os.path.join(FEATURE_PATH, EDGE_FEATURE_DIR, f"{sequence_name}.npy"))
                        max_dim_0 = min(min_len, full_edge_features_raw.shape[0]);
                        max_dim_1 = min(min_len, full_edge_features_raw.shape[1])
                        full_edge_features = full_edge_features_raw[:max_dim_0, :max_dim_1, :]
                        source_nodes, target_nodes = edge_index[0], edge_index[1]
                        edge_attr_np = full_edge_features[source_nodes.numpy(), target_nodes.numpy()]
                        edge_attr = torch.from_numpy(edge_attr_np.astype(np.float32))
                    except FileNotFoundError:
                        edge_attr = torch.zeros((edge_index.size(1), EDGE_DIM), dtype=torch.float32)
        except (FileNotFoundError, ValueError, IndexError):
            return None
        if node_features.shape[1] != IN_CHANNELS: return None
        return self.names[index], self.sequences[index], label, node_features, edge_index, edge_attr

    def __len__(self):
        return len(self.names)


def collate_fn(batch):
    batch = [item for item in batch if item is not None];
    return batch if batch else None


def train_one_epoch(model, data_loader):
    model.train();
    total_loss = 0.0
    for batch in data_loader:
        if not batch: continue
        data = batch[0];
        model.optimizer.zero_grad()
        _, _, labels, node_features, edge_index, edge_attr = data
        if node_features.shape[0] == 0: continue
        y_pred = model(torch.from_numpy(node_features), edge_index, edge_attr);
        y_true = torch.from_numpy(labels).long().to(device)
        if y_pred.shape[0] == 0 or y_pred.shape[0] != y_true.shape[0]: continue
        loss = model.criterion(y_pred, y_true);
        loss.backward();
        model.optimizer.step();
        total_loss += loss.item()
    return total_loss / len(data_loader) if data_loader and len(data_loader) > 0 else 0.0


def evaluate(model, data_loader):
    model.eval();
    total_loss, all_preds_list, all_labels_list = 0.0, [], []
    with torch.no_grad():
        for batch in data_loader:
            if not batch: continue
            data = batch[0];
            _, _, labels, node_features, edge_index, edge_attr = data
            if node_features.shape[0] == 0: continue
            y_pred = model(torch.from_numpy(node_features), edge_index, edge_attr);
            y_true = torch.from_numpy(labels).long().to(device)
            if y_pred.shape[0] == 0 or y_pred.shape[0] != y_true.shape[0]: continue
            loss = nn.CrossEntropyLoss()(y_pred, y_true);
            total_loss += loss.item()
            all_preds_list.append(F.softmax(y_pred, dim=1)[:, 1].cpu().numpy());
            all_labels_list.append(labels)
    avg_loss = total_loss / len(data_loader) if data_loader and len(data_loader) > 0 else 0.0
    all_labels = np.concatenate(all_labels_list) if all_labels_list else np.array([]);
    all_preds = np.concatenate(all_preds_list) if all_preds_list else np.array([])
    return avg_loss, all_labels, all_preds


def analysis(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.size == 0 or y_pred.size == 0 or len(set(y_true)) < 2: return {'f1': 0, 'AUC': 0.5, 'AUPRC': 0.0,
                                                                             'threshold': 0.5}
    best_f1, best_threshold = -1.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        binary_pred = (y_pred >= t).astype(int);
        f1 = metrics.f1_score(y_true, binary_pred, zero_division=0)
        if f1 > best_f1: best_f1, best_threshold = f1, t
    try:
        auc = metrics.roc_auc_score(y_true, y_pred); auprc = metrics.average_precision_score(y_true, y_pred)
    except ValueError:
        auc, auprc = 0.5, 0.0
    return {'f1': best_f1, 'AUC': auc, 'AUPRC': auprc, 'threshold': best_threshold}


def train(model, train_df, valid_df, fold):
    train_loader = DataLoader(ProDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn);
    valid_loader = DataLoader(ProDataset(valid_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=NUMBER_EPOCHS, eta_min=1e-6);
    best_auprc, best_epoch, patience_counter = 0.0, 0, 0
    print(f"\n========== [Fold {fold}] Training Start ==========")
    for epoch in range(1, NUMBER_EPOCHS + 1):
        start_time = time.time();
        train_loss = train_one_epoch(model, train_loader)
        valid_loss, valid_labels, valid_preds = evaluate(model, valid_loader);
        valid_results = analysis(valid_labels, valid_preds);
        scheduler.step()
        print(f"--- Epoch {epoch:03d}/{NUMBER_EPOCHS} --- Time: {time.time() - start_time:.2f}s ---")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"  Valid AUPRC: {valid_results['AUPRC']:.4f} (Best: {best_auprc:.4f}) | F1: {valid_results['f1']:.4f}")
        if valid_results['AUPRC'] > best_auprc:
            best_auprc, best_epoch = valid_results['AUPRC'], epoch;
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'Fold{fold}_best_model.pkl'));
            print(f"  🎉 New best model saved!");
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE: print(f"  Early stopping triggered."); break
    print(f"========== [Fold {fold}] Finished. Best AUPRC: {best_auprc:.4f} at epoch {best_epoch}. ==========")
    # ... (evaluation after training logic) ...
    if os.path.exists(os.path.join(MODEL_PATH, f'Fold{fold}_best_model.pkl')):
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'Fold{fold}_best_model.pkl')));
        _, final_labels, final_preds = evaluate(model, valid_loader);
        final_results = analysis(final_labels, final_preds)
        return best_epoch, final_results['AUC'], final_results['AUPRC']
    else:
        return 0, 0.5, 0.0


# --- 5. 主执行函数 ---
def cross_validation(all_df, fold_num=5):
    # ... (更新模型和打印信息) ...
    print("========== Cross-Validation Start ==========");
    print(f"Model: Parallel GTG-Style");
    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=SEED);
    results = []
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(all_df)):
        model = GraphPPIS_Parallel_GTG().to(device)
        results.append(train(model, all_df.iloc[train_idx], all_df.iloc[valid_idx], fold + 1))
    epochs, aucs, auprcs = zip(*results)
    print("\n\n=============== CV Summary ===============");
    print(
        f"Avg Best Epoch: {np.mean(epochs):.0f}, Avg AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}, Avg AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    return round(np.mean(epochs)) if epochs else 0


def train_full_model(all_df, avg_epoch):
    # ... (更新模型和打印信息) ...
    if avg_epoch <= 0: return
    print(f"\n\n========= Training Full Model (Parallel GTG-Style) for {avg_epoch} Epochs... =========")
    model = GraphPPIS_Parallel_GTG().to(device)
    loader = DataLoader(ProDataset(all_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    for epoch in range(1, avg_epoch + 1):
        loss = train_one_epoch(model, loader);
        print(f"--- Full Train Epoch {epoch}/{avg_epoch}, Loss: {loss:.4f} ---")
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'Full_model_parallel_gtg.pkl'))
    print(f"\n✅ Final Parallel GTG model saved to: {os.path.join(MODEL_PATH, 'Full_model_parallel_gtg.pkl')}")


def main():
    # ... (main 函数保持不变) ...
    if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
    with open(os.path.join(DATASET_PATH, "Train_335.pkl"), "rb") as f: data_dict = pickle.load(f)
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['sequence', 'label']);
    df.reset_index(inplace=True);
    df.rename(columns={'index': 'ID'}, inplace=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    avg_best_epoch = cross_validation(df)
    # train_full_model(df, avg_best_epoch)


if __name__ == "__main__":
    main()