"""Smoke test: verify NLP model and Claude agent are reachable."""
# Model Task / I/O - Input: A batch of RGB face images x with shape [B, 3, H, W] (typically resized/cropped to 224×224 for timm backbones). - Output: For each image, the model outputs 40 real-valued logits z with shape [B, 40], one logit per semantic attribute. - Post-processing: Apply sigmoid to logits: p = σ(z). Threshold probabilities (e.g., p > 0.5) to produce 40 binary attribute predictions per image. - Training Objective: Multi-label classification (each image may have multiple attributes true). Use BCEWithLogitsLoss between the 40 logits and the 40-dimensional binary target vector. --- Dataset / Data Loading - Dataset: CelebA (~200k celebrity face images) with 40 binary attribute annotations per image. Use the official train / val / test split provided by CelebA. - Each Sample Provides: image: RGB face photo; target: 40-dimensional binary vector y ∈ {0,1}^40. - The 40 Attributes: [["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs"],["Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair"],["Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair"],["Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes"],["No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks"],["Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat"],["Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]]
import json
import subprocess
import sys

NLP_MODEL = "opus"
AGENT_MODEL = "opus"
WORKING_SPACE = "/home/xuanhe_linux_001/agentic_probe_rein/dummy_project"


def test_nlp():
    print("── NLP model (Claude, no tools) ───────────────────")
    prompt = 'Reply with exactly this JSON and nothing else: {"status": "ok", "model": "nlp"}'
    result = subprocess.run(
        ["claude", "-p", "--model", NLP_MODEL, "--tools", "", "--no-session-persistence", prompt],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL — exit code {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        return
    data = json.loads(result.stdout)
    assert data.get("status") == "ok", f"Unexpected response: {data}"
    print(f"  PASS — got: {data}")


def test_agent():
    print("── Agent (Claude, full tools) ─────────────────────")
    result = subprocess.run(
        ["claude", "-p", "--model", AGENT_MODEL, "Reply with exactly the word PONG and nothing else."],
        cwd=WORKING_SPACE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL — exit code {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        sys.exit(1)
    output = result.stdout.strip()
    assert "PONG" in output, f"Unexpected response: {output!r}"
    print(f"  PASS — got: {output!r}")


def test_web_search():
    print("── Web search (NLP, CRWV stock price) ─────────────")
    prompt = (
        "Use WebSearch to find the current stock price of CRWV right now. "
        "Your entire response must be one single JSON object and absolutely nothing else — "
        "no prose, no markdown, no source list, no newlines after the closing brace. "
        'Format: {"ticker": "CRWV", "price": <number or null if not found>, "source": "<where you found it>"}'
    )
    result = subprocess.run(
        ["claude", "-p", "--model", NLP_MODEL, "--tools", "WebSearch,WebFetch",
         "--no-session-persistence", prompt],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL — exit code {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        return
    try:
        data = json.loads(result.stdout)
        assert data.get("ticker") == "CRWV", f"Unexpected response: {data}"
        print(f"  PASS — CRWV price: {data.get('price')} (source: {data.get('source')})")
    except json.JSONDecodeError:
        print(f"  FAIL — response was not JSON: {result.stdout.strip()!r}")


if __name__ == "__main__":
    try:
        test_nlp()
    except Exception as e:
        print(f"  FAIL — {e}")

    try:
        test_agent()
    except Exception as e:
        print(f"  FAIL — {e}")

    try:
        test_web_search()
    except Exception as e:
        print(f"  FAIL — {e}")
