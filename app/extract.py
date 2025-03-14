import numpy as np
import pandas as pd
import shogi


def extract_features(board: shogi.Board, side: str) -> dict:
    """
    盤面の特徴量を抽出する関数
    Args:
        board (shogi.Board): 盤面情報
        side (str): 評価対象の色
    Returns:
        dict: 引数 board から抽出した特徴量
    """
    features = {
        "loc": np.zeros((14, 9, 9), dtype=np.int8),
        "opp_loc": np.zeros((14, 9, 9), dtype=np.int8),
        "hand": np.zeros(7, dtype=np.int8),
        "opp_hand": np.zeros(7, dtype=np.int8),
    }

    # 各マスにある駒を特徴量に変換
    for i in range(9):
        for j in range(9):
            if side == shogi.BLACK:
                piece = board.piece_at(i * 9 + j)
            else:
                piece = board.piece_at((8 - i) * 9 + (8 - j))
            if piece is not None:
                if piece.color == side:
                    features["loc"][piece.piece_type - 1, i, j] = 1
                else:
                    features["opp_loc"][piece.piece_type - 1, i, j] = 1
    # 持ち駒情報を追加
    if side == shogi.BLACK:
        hand = board.pieces_in_hand[shogi.BLACK]
        opp_hand = board.pieces_in_hand[shogi.WHITE]
    else:
        hand = board.pieces_in_hand[shogi.WHITE]
        opp_hand = board.pieces_in_hand[shogi.BLACK]

    for piece_type, count in hand.items():
        features["hand"][piece_type - 1] = count

    for piece_type, count in opp_hand.items():
        features["opp_hand"][piece_type - 1] = count

    features = {key: value.tolist() for key, value in features.items()}
    return features
