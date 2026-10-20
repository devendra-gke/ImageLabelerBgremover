
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv(
    "HF_SEGMENTATION_MODEL",
    os.getenv("HF_IMAGE_MODEL", "briaai/RMBG-1.4"),
)

API_URL = f"https://huggingface.co/api/models/{MODEL_ID}"


def check_model():
    response = requests.get(
        API_URL,
        params={
            "expand[]": [
                "pipeline_tag",
                "inferenceProviderMapping",
                "library_name",
            ]
        },
        timeout=30,
    )

    if response.status_code == 404:
        print(f"Model repository not found: {MODEL_ID}")
        print("Check the exact model ID on Hugging Face.")
        return

    response.raise_for_status()
    data = response.json()

    print("Repository:", data.get("id"))
    print("Pipeline task:", data.get("pipeline_tag"))
    print("Library:", data.get("library_name"))
    print(
        "Inference provider mapping:",
        json.dumps(
            data.get("inferenceProviderMapping", {}),
            indent=2,
        ),
    )

    providers = data.get("inferenceProviderMapping") or {}
    pipeline_tag = (data.get("pipeline_tag") or "").strip().lower()

    if pipeline_tag and pipeline_tag != "image-segmentation":
        print(
            "\nWarning: this model is not tagged as image-segmentation. "
            "Background removal quality may be poor."
        )

    if not providers:
        print(
            "\nNo Inference Provider mapping was returned. "
            "This does not prove that no other deployment option exists."
        )
    else:
        print("\nCheck the listed providers for image input support.")
        print(
            "For background removal, verify that the model supports "
            "segmentation rather than only image generation."
        )


if __name__ == "__main__":
    check_model()
