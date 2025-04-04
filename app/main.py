from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.predict import predict_next_board

app = FastAPI()


class Data(BaseModel):
    currentSfen: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 任意のオリジンを許可（開発環境では "*" を使用）
    allow_credentials=True,
    allow_methods=["*"],  # 任意のメソッドを許可
    allow_headers=["*"],  # 任意のヘッダーを許可
)


@app.post("/next")
async def post_data(data: Data):
    try:
        sfen = data.currentSfen
        next_sfen, status = predict_next_board(sfen)
        # 正常レスポンス
        return {"message": next_sfen, "status": status}

    except Exception as e:
        # 予期しないエラー
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
