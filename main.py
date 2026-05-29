import cv2

from methods import proposed_ace, linear_um, cubic_um, rational_um, os_laplacian_enhancement
from evaluations import calculate_vr, calculate_nar

def process_and_compare(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if orig is None: return
    
    methods = {
        "Proposed ACE (YENI)": proposed_ace,
        "Linear UM": linear_um,
        "Cubic UM": cubic_um,
        "OS Laplacian": os_laplacian_enhancement,
        "Rational UM": rational_um,
    }

    results = {"Original": orig}
    evaluations = []

    for name, method in methods.items():
        enhanced = method(orig)
        results[name] = enhanced
        evaluations.append(
            (
                name,
                calculate_vr(orig, enhanced),
                calculate_nar(orig, method),
            )
        )
    
    for name, img in results.items():
        cv2.imwrite(f"{name.replace(' ', '_')}.png", img)
    print("Đã xuất các file so sánh.")

    print("Đánh giá:")
    for name, vr, nar in evaluations:
        print(f"- {name}: VR = {vr:.4f}, NAR = {nar:.4f}")


def main():
    """Điểm vào chương trình: đọc ảnh đầu vào và xuất các ảnh so sánh."""
    process_and_compare("image.png")


if __name__ == "__main__":
    main()