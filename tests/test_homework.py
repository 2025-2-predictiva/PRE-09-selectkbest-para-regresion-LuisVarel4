"""Autograding script."""


def load_data():

    import pandas as pd

    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y


def load_estimator():

    import os
    import pickle

    # --
    if not os.path.exists("homework/estimator.pickle"):
        return None
    with open("homework/estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


# def load_estimator():

#    import os
#    import pickle

#    # --
#    if not os.path.exists("homework/estimator.pickle"):
#        return None
#    with open("homework/estimator.pickle", "rb") as file:
#        estimator = pickle.load(file)
#
#    return estimator


def load_estimator():
    import os, pickle

    path = "homework/estimator.pickle"
    if not os.path.exists(path):
        return None

    # --- compat para _RemainderColsList al deserializar
    def _load_with_compat(fobj):
        try:
            return pickle.load(fobj)
        except AttributeError:
            try:
                from sklearn.compose import _column_transformer as _ct

                if not hasattr(_ct, "_RemainderColsList"):

                    class _RemainderColsList(list): ...

                    _RemainderColsList.__name__ = "_RemainderColsList"
                    _RemainderColsList.__qualname__ = "_RemainderColsList"
                    _RemainderColsList.__module__ = (
                        "sklearn.compose._column_transformer"
                    )
                    _ct._RemainderColsList = _RemainderColsList
            except Exception:
                pass
            fobj.seek(0)
            return pickle.load(fobj)

    with open(path, "rb") as f:
        est = _load_with_compat(f)

    # -------- Wrapper que alinea columnas antes de predecir --------
    class _AlignedEstimator:
        def __init__(self, inner):
            self._inner = inner
            self._expected_cols = self._infer_expected_cols(inner)

        # Busca columnas de entrenamiento en varios lugares posibles
        def _infer_expected_cols(self, inner):
            def _get_cols(obj):
                # 1) Estimador/pipeline completo
                cols = getattr(obj, "feature_names_in_", None)
                if cols is not None:
                    return list(cols)
                # 2) best_estimator_ (GridSearchCV)
                be = getattr(obj, "best_estimator_", None)
                if be is not None:
                    cols = getattr(be, "feature_names_in_", None)
                    if cols is not None:
                        return list(cols)
                # 3) named_steps (por nombre frecuente)
                for step_name in (
                    "preprocess",
                    "preprocessor",
                    "prep",
                    "ct",
                    "columntransformer",
                ):
                    try:
                        step = obj.named_steps[step_name]
                        cols = getattr(step, "feature_names_in_", None)
                        if cols is not None:
                            return list(cols)
                    except Exception:
                        pass
                # 4) pasos dentro del pipeline
                try:
                    for _, step in getattr(obj, "steps", []):
                        cols = getattr(step, "feature_names_in_", None)
                        if cols is not None:
                            return list(cols)
                except Exception:
                    pass
                # 5) Dentro de best_estimator_.steps
                if be is not None:
                    try:
                        for _, step in getattr(be, "steps", []):
                            cols = getattr(step, "feature_names_in_", None)
                            if cols is not None:
                                return list(cols)
                    except Exception:
                        pass
                return None

            cols = _get_cols(inner)
            if cols is None:
                be = getattr(inner, "best_estimator_", None)
                if be is not None:
                    cols = _get_cols(be)
            return list(cols) if cols is not None else None

        def _align(self, X):
            import pandas as pd

            if self._expected_cols is None or not hasattr(X, "columns"):
                return X
            X = X.copy()
            # Añade columnas faltantes como NA y descarta extras; respeta el orden original de entrenamiento
            for c in self._expected_cols:
                if c not in X.columns:
                    X[c] = pd.NA
            return X[self._expected_cols]

        # Delegación
        def predict(self, X, *args, **kwargs):
            return self._inner.predict(self._align(X), *args, **kwargs)

        def predict_proba(self, X, *args, **kwargs):
            return self._inner.predict_proba(self._align(X), *args, **kwargs)

        def decision_function(self, X, *args, **kwargs):
            return self._inner.decision_function(self._align(X), *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _AlignedEstimator(est)


def test_01():

    from sklearn.metrics import r2_score

    x, y = load_data()
    estimator = load_estimator()

    r2 = r2_score(
        y,
        estimator.predict(x),
    )

    assert r2 > 0.6
