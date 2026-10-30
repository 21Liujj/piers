#!/usr/bin/env python3
import argparse
import json

from model.text_semantic import TextPrototypeBank


def main() -> None:
    parser = argparse.ArgumentParser("Demo TextPrototypeBank")
    parser.add_argument(
        "--class_names",
        type=str,
        default='["road","sidewalk","building","person","car"]',
        help="JSON list of class names.",
    )
    parser.add_argument("--text_model", type=str, default="ViT-B-32")
    parser.add_argument("--text_pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    class_names = json.loads(args.class_names)
    bank = TextPrototypeBank(
        class_names=class_names,
        text_model=args.text_model,
        text_pretrained=args.text_pretrained,
        device=args.device,
    )
    protos = bank.refresh()
    print(f"[OK] built text prototypes: shape={tuple(protos.shape)}")
    print(f"classes={class_names}")


if __name__ == "__main__":
    main()
