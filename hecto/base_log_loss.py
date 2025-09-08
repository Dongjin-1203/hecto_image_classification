import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

def multiclass_log_loss(answer_df, submission_df):
    class_list = sorted(answer_df['label'].unique())
    
    if submission_df.shape[0] != answer_df.shape[0]:
        raise ValueError("submission_df 행 개수가 answer_df와 일치하지 않습니다.")

    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
    answer_df = answer_df.sort_values(by='ID').reset_index(drop=True)

    if not all(answer_df['ID'] == submission_df['ID']):
        raise ValueError("ID가 정렬되지 않았거나 불일치합니다.")
    
    missing_cols = [col for col in class_list if col not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"클래스 컬럼 누락: {missing_cols}")
    
    if submission_df[class_list].isnull().any().any():
        raise ValueError("NaN 포함됨")
    for col in class_list:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            raise ValueError(f"{col}의 확률값이 0~1 범위 초과")

    # 정답 인덱스 변환
    true_labels = answer_df['label'].tolist()
    true_idx = [class_list.index(lbl) for lbl in true_labels]

    # 확률 정규화 + clip
    probs = submission_df[class_list].values
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.clip(probs, 1e-15, 1 - 1e-15)

    return log_loss(true_idx, y_pred, labels=list(range(len(class_list))))

# 예시 데이터 (클래스 3개: A, B, C)
answer_df = pd.DataFrame({
    'ID': [1, 2, 3],
    'label': ['A', 'B', 'C']
})

submission_df = pd.DataFrame({
    'ID': [1, 2, 3],
    'A': [0.7, 0.1, 0.2],
    'B': [0.2, 0.8, 0.3],
    'C': [0.1, 0.1, 0.5]
})

# 함수 실행
loss = multiclass_log_loss(answer_df, submission_df)
print(f"Log Loss: {loss:.6f}")