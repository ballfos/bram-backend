import shogi
import torch
import torch.nn as nn

from app.extract import extract_features
from app.model import ShogiModel


def predict_next_board(current_sfen: str) -> str:

    # モデルのインスタンス作成
    model = ShogiModel()
    # モデルのパス
    path_to_model = "./model.pth"
    # モデルをロード
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))

    # モデルを評価モードに設定
    model.eval()

    board = shogi.Board(current_sfen)

    # sfenから先手後手の取得
    side = board.turn
    # 合法手のリスト化
    legal_moves = list(board.legal_moves)

    best_probability = 0
    for legal_move in legal_moves:

        board.push(legal_move)
        # 特徴抽出
        feature = extract_features(board, side)
        # リスト形式から Tensor に変換
        feature["loc"] = torch.tensor(feature["loc"], dtype=torch.float32)
        feature["opp_loc"] = torch.tensor(feature["opp_loc"], dtype=torch.float32)
        feature["hand"] = torch.tensor(feature["hand"], dtype=torch.float32)
        feature["opp_hand"] = torch.tensor(feature["opp_hand"], dtype=torch.float32)
        loc = feature["loc"].unsqueeze(0)  # バッチ次元を追加
        opp_loc = feature["opp_loc"].unsqueeze(0)
        hand = feature["hand"].unsqueeze(0)
        opp_hand = feature["opp_hand"].unsqueeze(0)
        # 確率予測
        output = model(loc, opp_loc, hand, opp_hand)
        # 確率の高い手を記憶
        if best_probability < output:
            best_probability = output
            next_sfen = board.sfen()

        board.pop()
    return next_sfen
