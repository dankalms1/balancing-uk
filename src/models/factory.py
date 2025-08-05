from .SA_BiLSTM import SA_BiLSTM
from .BiAttnPointForecaster import BiAttnPointForecaster

MODEL_FACTORY = {
    "SA_BiLSTM": SA_BiLSTM,
    "BiAttnPointForecaster": BiAttnPointForecaster,
}
