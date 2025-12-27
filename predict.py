# Simple test model - no GPU needed
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Setup complete!")

    def predict(
        self,
        text: str = Input(description="Test input", default="hello"),
    ) -> str:
        return f"Echo: {text} - Token works!"
