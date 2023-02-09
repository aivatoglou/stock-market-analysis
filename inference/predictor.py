import onnxruntime as ort


class Predictor:
    def __init__(self, full_path, provider) -> None:

        # Load session for inference
        self.ort_session = ort.InferenceSession(full_path, providers=provider)

    def __call__(self):
        pass

    def predict(self, input_matrix):
        output = self.ort_session.run(None, {"input": input_matrix})
        return output[0]
