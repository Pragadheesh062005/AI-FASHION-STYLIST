import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from deepface import DeepFace



# ----------------------
# Helpers
# ----------------------
def pil_to_cv(img_pil: Image.Image):
    """PIL -> OpenCV BGR"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray):
    """OpenCV BGR -> PIL"""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def extract_top_colors(img_pil: Image.Image, n_colors=5):
    """
    Extract a color palette from the clothing area (heuristic: lower 70% of the image)
    using KMeans on a resized copy for speed.
    Returns hex colors sorted by cluster size.
    """
    img = np.array(img_pil)
    h, w, _ = img.shape

    # Heuristic: focus more on clothing area (exclude top 30% face area)
    start_y = int(0.30 * h)
    clothing = img[start_y:h, :, :]

    # Resize for speed
    small = cv2.resize(clothing, (min(300, clothing.shape[1]), int(min(300, clothing.shape[0]))), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3)

    # KMeans
    k = min(n_colors, len(data))
    if k < 1:
        return []

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Sort by cluster frequency
    counts = np.bincount(labels)
    order = np.argsort(-counts)

    palette = []
    for idx in order:
        r, g, b = centers[idx]
        palette.append('#{:02x}{:02x}{:02x}'.format(r, g, b))
    return palette

def complementary(hex_color):
    """Return a simple complementary color in hex (180° hue flip approximation)."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return "#999999"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    comp = (255 - r, 255 - g, 255 - b)
    return '#{:02x}{:02x}{:02x}'.format(*comp)

def suggest_outfits(gender: str, palette: list[str]):
    """
    Rule-based suggestions by occasion, informed by the user's palette.
    gender: 'Man' or 'Woman' (capitalized for display)
    """
    # Choose a lead color (largest cluster)
    lead = palette[0] if palette else "#2e2e2e"
    accent = complementary(lead)

    # Basic text blocks per gender (expand freely)
    if gender.lower().startswith('w'):  # Woman
        party = [
            f"Sequin or satin dress in {lead} with {accent} accessories.",
            f"Statement heels and clutch in {accent}.",
            "Soft waves or sleek ponytail; bold lip."
        ]
        formal = [
            f"Tailored pantsuit or sheath dress in {lead}.",
            f"Minimal jewelry; pumps in {accent} or nude.",
            "Structured tote, neutral makeup."
        ]
        casual = [
            f"High-waist jeans + relaxed tee in {lead}.",
            f"Layer with a cropped jacket; sneakers in {accent}.",
            "Simple studs, light makeup."
        ]
    else:  # Man
        party = [
            f"Unstructured blazer in {lead} over a minimal tee.",
            f"Chelsea boots or clean sneakers; watch with {accent} strap.",
            f"Slim trousers; pocket square with hint of {accent}."
        ]
        formal = [
            f"Two-piece suit in {lead} or charcoal.",
            f"Shirt in crisp white; tie with {accent} micro-pattern.",
            "Oxford shoes, leather belt."
        ]
        casual = [
            f"Chinos + henley or crew tee in {lead}.",
            f"Lightweight overshirt; sneakers with {accent} detail.",
            "Cap or bracelet optional."
        ]

    # Return structured suggestions
    return {
        "Party": party,
        "Formal": formal,
        "Casual": casual,
        "Colors": {
            "Lead (from your outfit)": lead,
            "Complement": accent,
            "Palette": palette[:5]
        }
    }

def detect_gender(img_pil: Image.Image) -> str:
    """
    Use DeepFace to analyze gender. Returns 'Man' or 'Woman'.
    Compatible with all DeepFace versions.
    """
    img = np.array(img_pil)

    try:
        # Newer versions (>=0.0.92)
        result = DeepFace.analyze(
            img_path=img,
            actions=['gender'],
            detector_backend='retinaface',
            enforce_detection=True
        )
    except TypeError:
        # Older versions: no prog_bar param allowed
        result = DeepFace.analyze(
            img_path=img,
            actions=['gender'],
            detector_backend='retinaface',
            enforce_detection=True
        )
    except Exception:
        # Fallback without strict detection
        result = DeepFace.analyze(
            img_path=img,
            actions=['gender'],
            enforce_detection=False
        )

    if isinstance(result, list) and len(result) > 0:
        gender_str = result[0].get("dominant_gender", "Man")
    else:
        gender_str = result.get("dominant_gender", "Man")

    return "Woman" if gender_str.lower().startswith("w") else "Man"

def render_palette(palette):
    cols = st.columns(len(palette)) if palette else []
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #ddd;height:60px;background:{palette[i]};border-radius:8px"></div>
                <div style="text-align:center;margin-top:6px;font-size:12px">{palette[i]}</div>
                """,
                unsafe_allow_html=True
            )

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="AI Fashion Stylist", page_icon="🧥", layout="wide")
st.title("🧥 AI Fashion Stylist")
st.write("Upload your photo and get personalized outfit suggestions for **Party**, **Formal**, and **Casual**.")

with st.sidebar:
    st.header("Settings")
    n_colors = st.slider("Palette size", 3, 7, 5)
    st.caption("Tip: clear, well-lit photos work best.")

uploaded = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_bytes = uploaded.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.subheader("Preview")
    st.image(img_pil, use_column_width=True)

    with st.spinner("Analyzing photo..."):
        # 1) Gender
        gender = detect_gender(img_pil)

        # 2) Colors
        palette = extract_top_colors(img_pil, n_colors=n_colors)

        # 3) Suggestions
        suggestions = suggest_outfits(gender, palette)

    st.success(f"Detected: **{gender}**")
    st.divider()

    st.subheader("Your Color Palette")
    if palette:
        render_palette(palette)
    else:
        st.info("Couldn’t extract colors—try a different photo or brighter lighting.")

    st.divider()
    st.subheader("Suggestions")

    for occasion in ["Party", "Formal", "Casual"]:
        with st.expander(f"{occasion}", expanded=(occasion=="Party")):
            for tip in suggestions[occasion]:
                st.markdown(f"- {tip}")

    st.divider()
    st.subheader("Styling Colors")
    cols = suggestions["Colors"]
    st.markdown(f"**Lead (from your outfit):** `{cols['Lead (from your outfit)']}`  |  **Complement:** `{cols['Complement']}`")
    if cols["Palette"]:
        st.write("**Palette:**", ", ".join(cols["Palette"]))

    st.caption("Disclaimer: This is a demo stylist. Always dress for comfort, weather, and cultural context.")

else:
    st.info("Upload a clear photo (waist-up or selfie) to begin.")

