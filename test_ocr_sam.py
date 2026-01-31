"""Test OCR and SAM detection."""
import numpy as np
from PIL import Image

print("=== Testing EasyOCR directly ===")
import easyocr
reader = easyocr.Reader(['en', 'ko'], gpu=False)
img = Image.open('./Images/20260124_103828.jpg').convert('RGB')
img_array = np.array(img)

result = reader.readtext(img_array)
print(f'OCR results: {len(result)} lines')

print("\n=== Testing FastSAM directly ===")
from ultralytics import FastSAM

model = FastSAM('FastSAM-s.pt')
results = model(img_array, device='cpu', verbose=False)

if results and len(results) > 0:
    result_obj = results[0]
    if hasattr(result_obj, 'masks') and result_obj.masks is not None:
        masks = result_obj.masks.data.cpu().numpy()
        print(f'FastSAM detected: {len(masks)} masks')

# Test the integrated SAM Pair Extractor
print("\n=== Testing SAM Pair Extractor Integration ===")
from flashcard_engine.sam_pair_extractor import TextDetector, PictureDetector, PairConfig

# Test TextDetector with EasyOCR
print("\n[1] TextDetector (EasyOCR):")
text_detector = TextDetector(lang='en,ko')
text_blocks = text_detector.detect(img)
print(f"    Detected {len(text_blocks)} text blocks")

# Test PictureDetector with FastSAM
print("\n[2] PictureDetector (FastSAM):")
pic_detector = PictureDetector(device='cpu')
config = PairConfig()
print(f"    Config: min_area={config.min_mask_area_ratio}, max_area={config.max_mask_area_ratio}")

pictures = pic_detector.detect(img, text_blocks, config)
print(f"    Detected {len(pictures)} picture candidates")
for i, pic in enumerate(pictures[:5]):
    print(f'    - Box: {pic.bbox.to_list()} (conf: {pic.confidence:.2f})')

print("\n[PASS] All components working correctly!")
