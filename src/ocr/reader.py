"""
OCR text extraction from images and PDFs.

This module provides the interface to OCR engines (Tesseract, EasyOCR, AI Vision).
Currently implements placeholder functions that will be replaced with actual OCR engines.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import logging
import base64
import io
import json

from .models import OCRMetadata, TextBlock
from PIL import Image

logger = logging.getLogger(__name__)

# Try to load .env file for API keys and model configuration
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Import model configuration from config module
    try:
        import sys
        from pathlib import Path
        # Add parent directory (src/) to path to import config
        config_path = Path(__file__).parent.parent
        if str(config_path) not in sys.path:
            sys.path.insert(0, str(config_path))
        from config import IMAGE_MODEL
        logger.debug(f"[OCR] Loaded IMAGE_MODEL from config: {IMAGE_MODEL}")
    except ImportError:
        # Fallback if config module not available - read directly from env
        IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-4o")
        logger.debug(f"[OCR] Using IMAGE_MODEL from env (config module not available): {IMAGE_MODEL}")
except ImportError:
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-4o")
    logger.debug("[OCR] python-dotenv not available, Vision API key must be set via environment variable")
except Exception as e:
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-4o")
    logger.debug(f"[OCR] Error loading .env: {e}")


def extract_text_from_image(
    image_path: Path,
    language: str = "eng",
    engine: str = "auto",
    enable_vision: bool = False
) -> Tuple[List[TextBlock], OCRMetadata]:
    """
    Extract text blocks from an image file using OCR.
    
    Args:
        image_path: Path to image file (PNG, JPG, JPEG)
        language: Language code for OCR (default: "eng")
        engine: OCR engine to use ("tesseract", "easyocr", "vision", "auto")
        enable_vision: Whether to enable Vision API as fallback (default: False)
    
    Returns:
        Tuple of (list of TextBlock objects, OCRMetadata)
    
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image format is unsupported
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Validate image format
    valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    if image_path.suffix not in valid_extensions:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    
    logger.warning(f"[OCR] Extracting text from image: {image_path} (engine: {engine})")
    
    # Get image dimensions
    dimensions = _get_image_dimensions(image_path)
    
    text_blocks: List[TextBlock] = []
    confidence = 0.0
    used_engine = "none"
    vision_model = None
    
    # Detect if handwritten from filename (mirrors ENG-101-table-detector line 2121)
    filename_lower = image_path.name.lower()
    is_handwritten = any(keyword in filename_lower for keyword in ['handwritten', 'hand', 'scribble', 'notes'])
    
    # DPI-aware preprocessing: upscale image if low DPI (mirrors ENG-101-table-detector DPI handling)
    # This improves OCR quality for images, addressing root cause of null fields
    original_image_path = image_path
    temp_file_created = False
    try:
        with Image.open(image_path) as img:
            img_upscaled = _upscale_image_if_low_dpi(img, min_dpi=200)
            if img_upscaled.size != img.size:
                # Save upscaled image temporarily for OCR
                import tempfile
                import os
                tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                img_upscaled.save(tmp_file.name, 'PNG')
                tmp_file.close()
                image_path = Path(tmp_file.name)
                temp_file_created = True
                logger.debug(f"[OCR] Using upscaled image for better OCR quality")
    except Exception as e:
        logger.debug(f"[OCR] Error upscaling image: {e}, using original")
    
    # Try OCR engines in order: For handwritten, EasyOCR FIRST (better for handwriting); otherwise Tesseract first
    # This mirrors ENG-101-table-detector behavior (line 1747-1768)
    if is_handwritten and (engine == "auto" or engine == "easyocr"):
        # For handwritten, try EasyOCR FIRST (it's much better for handwriting)
        try:
            text_blocks, confidence = _extract_with_easyocr(image_path, language, dimensions)
            if text_blocks:
                used_engine = "easyocr"
                logger.warning("[OCR] EasyOCR used (handwritten detected, tried first)")
        except Exception as e:
            logger.debug(f"[OCR] EasyOCR failed for handwritten: {e}")
    
    # Try Tesseract (with PSM fallbacks for handwritten if EasyOCR didn't work)
    if not text_blocks and (engine == "auto" or engine == "tesseract"):
        try:
            text_blocks, confidence = _extract_with_tesseract(
                image_path, language, dimensions, 
                is_handwritten=is_handwritten
            )
            if text_blocks:
                used_engine = "tesseract"
                logger.warning(f"[OCR] Tesseract completed: {len(text_blocks)} text blocks, avg confidence: {confidence:.1f}")
            else:
                logger.warning("[OCR] Tesseract returned no text blocks")
        except Exception as e:
            logger.warning(f"[OCR] Tesseract failed: {e}")
    
        # If handwritten and Tesseract failed or returned no blocks, try PSM fallbacks
        # This mirrors ENG-101-table-detector behavior (line 1788-1793)
        if is_handwritten and not text_blocks:
            logger.debug("[OCR] Trying Tesseract PSM fallbacks for handwritten...")
            # Try PSM 11 (sparse text), 12 (sparse with OSD), 13 (raw line)
            for psm_mode in [11, 12, 13]:
                try:
                    text_blocks, confidence = _extract_with_tesseract(
                        image_path, language, dimensions,
                        is_handwritten=is_handwritten,
                        psm_mode=psm_mode
                    )
                    if text_blocks:
                        used_engine = "tesseract"
                        logger.warning(f"[OCR] Tesseract PSM {psm_mode} succeeded for handwritten: {len(text_blocks)} text blocks")
                        break
                except Exception as e:
                    logger.debug(f"[OCR] Tesseract PSM {psm_mode} failed: {e}")
    
    # If Tesseract didn't work and NOT handwritten, try EasyOCR
    if not text_blocks and not is_handwritten and (engine == "auto" or engine == "easyocr"):
        try:
            text_blocks, confidence = _extract_with_easyocr(image_path, language, dimensions)
            if text_blocks:
                used_engine = "easyocr"
                logger.info("[OCR] EasyOCR used")
        except Exception as e:
            logger.debug(f"[OCR] EasyOCR failed: {e}")
    
    # If neither worked and Vision is enabled, try Vision API
    if not text_blocks and enable_vision and (engine == "auto" or engine == "vision"):
        try:
            text_blocks, confidence, vision_model = _extract_with_vision(image_path, dimensions)
            if text_blocks:
                used_engine = "vision"
                logger.info(f"[OCR] Using Vision API (model={vision_model})")
        except Exception as e:
            logger.debug(f"[OCR] Vision API failed: {e}")
    
    # If nothing worked, use fallback
    if not text_blocks:
        text_blocks, confidence = _extract_fallback_image(image_path, dimensions)
        used_engine = "fallback"
        logger.info("[OCR] Fallback used")
    
    # Clean up temporary upscaled image if created
    if temp_file_created and image_path != original_image_path:
        try:
            import os
            os.unlink(image_path)
        except Exception as e:
            logger.debug(f"[OCR] Error cleaning up temp file: {e}")
    
    metadata = OCRMetadata(
        engine=used_engine,
        language=language,
        confidence=confidence,
        page_number=1,
        page_count=1,
        image_dimensions=dimensions,
        vision_model=vision_model
    )
    
    logger.info(f"[OCR] Extracted {len(text_blocks)} text blocks from image")
    return text_blocks, metadata


def extract_text_from_pdf(
    pdf_path: Path,
    language: str = "eng",
    engine: str = "auto",
    page_range: Optional[Tuple[int, int]] = None,
    enable_vision: bool = False
) -> Tuple[List[TextBlock], OCRMetadata]:
    """
    Extract text blocks from a PDF document using OCR.
    
    Args:
        pdf_path: Path to PDF file
        language: Language code for OCR (default: "eng")
        engine: OCR engine to use ("tesseract", "easyocr", "vision", "auto")
        page_range: Optional tuple (start_page, end_page) to process specific pages
        enable_vision: Whether to enable Vision API as fallback (default: False)
    
    Returns:
        Tuple of (list of TextBlock objects, OCRMetadata)
    
    Raises:
        FileNotFoundError: If PDF file does not exist
        ValueError: If PDF is corrupted or unreadable
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"[OCR] Extracting text from PDF: {pdf_path} (engine: {engine})")
    
    # Try to extract text directly from PDF first (no OCR needed if text is embedded)
    text_blocks: List[TextBlock] = []
    page_count = 1
    confidence = 0.0
    used_engine = "none"
    pages_need_ocr = []
    
    # Step 1: Check for embedded text using PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)
            
            start_page = (page_range[0] - 1) if page_range else 0
            end_page = (page_range[1]) if page_range else page_count
            
            logger.debug(f"[OCR] PDF has {page_count} pages, extracting pages {start_page+1} to {end_page}")
            
            for page_num in range(start_page, min(end_page, page_count)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    # Text is embedded - extract as text blocks
                    text_blocks.extend(_parse_pdf_text_to_blocks(page_text, page_num + 1))
                    confidence = max(confidence, 0.9)  # High confidence for embedded text
                else:
                    # No embedded text - mark for OCR
                    pages_need_ocr.append(page_num)
    except ImportError:
        logger.debug("[OCR] PyPDF2 not available, will try OCR on all pages")
        pages_need_ocr = list(range(page_count))
    except Exception as e:
        logger.warning(f"[OCR] Error reading PDF: {e}, will try OCR")
        pages_need_ocr = list(range(page_count))
    
    # Step 2: For pages without embedded text, convert to images and run OCR
    vision_model = None
    if pages_need_ocr:
        logger.info(f"[OCR] PDF → image conversion needed for {len(pages_need_ocr)} pages")
        ocr_text_blocks, ocr_engine, ocr_vision_model = _extract_from_pdf_images(
            pdf_path, pages_need_ocr, language, engine, enable_vision
        )
        if ocr_text_blocks:
            text_blocks.extend(ocr_text_blocks)
            used_engine = ocr_engine
            vision_model = ocr_vision_model
            # Update confidence if OCR found text
            ocr_conf = sum(b.confidence for b in ocr_text_blocks) / len(ocr_text_blocks)
            confidence = max(confidence, ocr_conf)
        else:
            # No OCR results - try Vision as final fallback if enabled and results are poor
            if enable_vision and (len(text_blocks) < 5 or confidence < 0.5):
                try:
                    logger.info("[OCR] Trying Vision API as final fallback for PDF")
                    vision_blocks, vision_conf, vision_model = _extract_from_pdf_vision(
                        pdf_path, pages_need_ocr, dimensions=None
                    )
                    if vision_blocks:
                        text_blocks.extend(vision_blocks)
                        used_engine = "vision"
                        confidence = max(confidence, vision_conf)
                        logger.info(f"[OCR] Using Vision API (model={vision_model})")
                except Exception as e:
                    logger.debug(f"[OCR] Vision API fallback failed: {e}")
            
            # If still no results, check if we have any embedded text
            if not text_blocks:
                logger.debug("[OCR] No embedded text and OCR failed, using fallback")
                text_blocks, confidence = _extract_fallback_pdf(pdf_path, page_range)
                used_engine = "fallback"
            else:
                # We have embedded text, just no OCR results
                if used_engine == "none":
                    used_engine = "pypdf2"
    else:
        # All text was embedded
        used_engine = "pypdf2"
    
    metadata = OCRMetadata(
        engine=used_engine,
        language=language,
        confidence=confidence,
        page_number=1,
        page_count=page_count,
        vision_model=vision_model
    )
    
    logger.info(f"[OCR] Extracted {len(text_blocks)} text blocks from PDF ({page_count} pages)")
    return text_blocks, metadata


def _detect_ocr_engine() -> str:
    """
    Auto-detect available OCR engine.
    
    Returns:
        Engine name: "tesseract", "easyocr", "vision", or "none"
    """
    # Check for pytesseract + Tesseract binary
    try:
        import pytesseract
        try:
            pytesseract.get_tesseract_version()
            logger.debug("[OCR] Tesseract engine detected")
            return "tesseract"
        except Exception:
            pass
    except ImportError:
        pass
    
    # Check for easyocr
    try:
        import easyocr
        logger.debug("[OCR] EasyOCR engine detected")
        return "easyocr"
    except ImportError:
        pass
    
    # Check for AI Vision (would need API key check)
    # For now, skip this check to avoid external API calls in tests
    
    logger.debug("[OCR] No OCR engine detected, using fallback")
    return "none"


def _extract_with_tesseract(
    image_path: Path,
    language: str,
    dimensions: Optional[Tuple[int, int]],
    is_handwritten: bool = False,
    psm_mode: Optional[int] = None
) -> Tuple[List[TextBlock], float]:
    """
    Extract text using Tesseract OCR with preprocessing.
    
    Mirrors ENG-101-table-detector preprocessing behavior:
    - Handwritten: denoising + Otsu thresholding + light morphology
    - Printed: adaptive thresholding + dilation
    This improves OCR quality before extraction, addressing root cause of null fields.
    
    Args:
        image_path: Path to image file
        language: Language code for OCR
        dimensions: Image dimensions (width, height)
        is_handwritten: Whether image contains handwritten text
        psm_mode: Tesseract PSM mode (if None, uses default; for handwritten, try 11, 12, 13)
    """
    import pytesseract
    from PIL import Image
    
    img_original = Image.open(image_path)
    img = img_original
    
    # OCR PREPROCESSING (mirrors ENG-101-table-detector/src/extract_table_and_map_schema.py:1700-1741)
    # This improves OCR quality, which directly improves extraction accuracy for handwritten/PNG sources
    # If cv2/numpy not available, skip preprocessing and use original image
    img_preprocessed = None
    try:
        import cv2
        import numpy as np
        
        # Convert to numpy array for preprocessing
        img_array = np.array(img_original)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        if is_handwritten:
            # Handwritten-specific preprocessing (ENG-101: _preprocess_for_handwritten, line 1445)
            # Denoise first (handwritten text benefits from noise reduction)
            img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
            
            # Use Otsu's threshold instead of adaptive for handwritten (often works better)
            _, img_thresh = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Light morphological operations to smooth without losing detail
            kernel = np.ones((2, 2), np.uint8)
            img_processed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            img_preprocessed = Image.fromarray(img_processed)
            logger.warning(f"[OCR] Applied handwritten preprocessing (denoise + Otsu + morphology)")
        else:
            # Standard preprocessing for printed text (ENG-101: line 1735-1741)
            img_thresh = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # Apply light dilation (kernel 1x2) to connect characters slightly
            kernel = np.ones((1, 2), np.uint8)
            img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)
            img_preprocessed = Image.fromarray(img_dilated)
            logger.warning(f"[OCR] Applied printed text preprocessing (adaptive threshold + dilation)")
    except (ImportError, Exception) as e:
        # If preprocessing fails (cv2/numpy not available or processing error), use original image
        logger.warning(f"[OCR] Preprocessing skipped (cv2/numpy unavailable or error): {e}, using original image")
        img_preprocessed = None
    
    # Use preprocessed image if available, otherwise original
    img = img_preprocessed if img_preprocessed is not None else img_original
    
    # Build Tesseract config string
    # Default PSM: 6 (uniform block of text) for printed, or use provided psm_mode
    if psm_mode is not None:
        tesseract_config = f"--oem 3 --psm {psm_mode}"
    else:
        tesseract_config = "--oem 3 --psm 6"  # Default for printed text
    
    # Get OCR data with bounding boxes using image_to_data
    try:
        ocr_data = pytesseract.image_to_data(img, lang=language, config=tesseract_config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        # If preprocessed image fails and we have original, try original (especially for handwritten)
        if img_preprocessed is not None and is_handwritten:
            logger.debug(f"[OCR] Preprocessed image failed, trying original: {e}")
            ocr_data = pytesseract.image_to_data(img_original, lang=language, config=tesseract_config, output_type=pytesseract.Output.DICT)
        else:
            raise
    
    text_blocks = []
    current_line = 1
    last_y = None
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:  # Only non-empty text
            conf = float(ocr_data['conf'][i]) / 100.0 if ocr_data['conf'][i] != -1 else 0.5
            x = float(ocr_data['left'][i])
            y = float(ocr_data['top'][i])
            w = float(ocr_data['width'][i])
            h = float(ocr_data['height'][i])
            
            # Determine line number based on y-position
            line_num = None
            if last_y is None:
                line_num = current_line
                last_y = y
            elif abs(y - last_y) <= 10.0:  # Same line threshold
                line_num = current_line
            else:
                current_line += 1
                line_num = current_line
                last_y = y
            
            text_blocks.append(TextBlock(
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=conf,
                line_number=line_num,
                block_type="text"
            ))
    
    avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
    logger.warning(f"[OCR] Tesseract extracted {len(text_blocks)} text blocks (avg confidence: {avg_confidence:.1f})")
    return text_blocks, avg_confidence


def _extract_with_easyocr(
    image_path: Path,
    language: str,
    dimensions: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float]:
    """Extract text using EasyOCR."""
    import easyocr
    
    # Map language codes (EasyOCR uses 'en' not 'eng')
    lang_map = {"eng": "en", "en": "en"}
    easyocr_lang = lang_map.get(language, "en")
    
    reader = easyocr.Reader([easyocr_lang])
    results = reader.readtext(str(image_path))
    
    text_blocks = []
    current_line = 1
    last_y = None
    
    for (bbox, text, conf) in results:
        # EasyOCR bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Determine line number based on y-position
        line_num = None
        if last_y is None:
            line_num = current_line
            last_y = y_min
        elif abs(y_min - last_y) <= 10.0:  # Same line threshold
            line_num = current_line
        else:
            current_line += 1
            line_num = current_line
            last_y = y_min
        
        text_blocks.append(TextBlock(
            text=text.strip(),
            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
            confidence=float(conf),
            line_number=line_num,
            block_type="text"
        ))
    
    avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
    logger.debug(f"[OCR] EasyOCR extracted {len(text_blocks)} text blocks")
    return text_blocks, avg_confidence


def _extract_with_vision(
    image_path: Path,
    dimensions: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float, Optional[str]]:
    """
    Extract text using OpenAI Vision API.
    
    Args:
        image_path: Path to image file
        dimensions: Image dimensions (width, height)
    
    Returns:
        Tuple of (list of TextBlock objects, average confidence, vision model name)
    
    Raises:
        ValueError: If API key is not available
        Exception: If API call fails
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env file or environment variable.")
    
    try:
        from openai import OpenAI
        from PIL import Image
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Read and encode image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Use IMAGE_MODEL from config/env, with fallback to gpt-4-turbo if configured model fails
        vision_model = IMAGE_MODEL
        fallback_model = "gpt-4-turbo"  # Generic fallback for vision-capable models
        logger.info(f"[OCR Vision] Using model: {vision_model} (fallback: {fallback_model})")
        
        try:
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract all text from this image with bounding boxes. 
Return a JSON array where each object has:
- "text": the extracted text string
- "bbox": [x_min, y_min, x_max, y_max] in pixels
- "confidence": confidence score 0.0-1.0

Image dimensions: {}x{} pixels.

Return ONLY valid JSON, no markdown formatting.""".format(img_width, img_height)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
        except Exception as e:
            # Fallback to gpt-4-turbo if configured model fails (e.g., model not available)
            logger.warning(f"[OCR Vision] Model {vision_model} failed: {e}, trying fallback {fallback_model}")
            if vision_model != fallback_model:
                vision_model = fallback_model
                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract all text from this image with bounding boxes. 
Return a JSON array where each object has:
- "text": the extracted text string
- "bbox": [x_min, y_min, x_max, y_max] in pixels
- "confidence": confidence score 0.0-1.0

Image dimensions: {}x{} pixels.

Return ONLY valid JSON, no markdown formatting.""".format(img_width, img_height)
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000
                )
            else:
                raise
        
        # Parse response
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Parse JSON
        try:
            blocks_data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                blocks_data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from Vision API response: {content[:200]}")
        
        # Convert to TextBlock objects
        text_blocks = []
        current_line = 1
        last_y = None
        
        for block_data in blocks_data:
            if not isinstance(block_data, dict):
                continue
            
            text = block_data.get("text", "").strip()
            if not text:
                continue
            
            bbox = block_data.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            conf = float(block_data.get("confidence", 0.9))
            
            # Determine line number
            line_num = None
            if last_y is None:
                line_num = current_line
                last_y = y_min
            elif abs(y_min - last_y) <= 10.0:
                line_num = current_line
            else:
                current_line += 1
                line_num = current_line
                last_y = y_min
            
            text_blocks.append(TextBlock(
                text=text,
                bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
                confidence=conf,
                line_number=line_num,
                block_type="text"
            ))
        
        avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
        logger.debug(f"[OCR] Vision API extracted {len(text_blocks)} text blocks")
        return text_blocks, avg_confidence, vision_model
    
    except ImportError:
        raise ValueError("openai package not installed. Install with: pip install openai")
    except Exception as e:
        logger.warning(f"[OCR] Vision API extraction failed: {e}")
        raise


def _extract_fallback_image(
    image_path: Path,
    dimensions: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float]:
    """
    Fallback text extraction for images when no OCR engine is available.
    
    Returns empty blocks for test compatibility.
    """
    logger.debug("[OCR] Using fallback image extraction (no OCR engine)")
    return [], 0.0


def _extract_fallback_pdf(
    pdf_path: Path,
    page_range: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float]:
    """
    Fallback text extraction for PDFs when no text extraction is available.
    
    Returns empty blocks for test compatibility.
    """
    logger.debug("[OCR] Using fallback PDF extraction (no text extraction available)")
    return [], 0.0


def _parse_pdf_text_to_blocks(text: str, page_number: int) -> List[TextBlock]:
    """
    Parse PDF extracted text into TextBlock objects.
    
    Uses simple line-based parsing with estimated positions.
    """
    if not text.strip():
        return []
    
    lines = text.split('\n')
    text_blocks = []
    y_offset = 50.0  # Starting y position
    line_height = 20.0
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if line:
            # Estimate bounding box (simple heuristic)
            x_start = 50.0
            x_end = x_start + len(line) * 8.0  # Rough character width estimate
            y_start = y_offset + (line_num * line_height)
            y_end = y_start + line_height
            
            text_blocks.append(TextBlock(
                text=line,
                bbox=(x_start, y_start, x_end, y_end),
                confidence=0.9,  # High confidence for embedded text
                line_number=line_num + 1,
                block_type="text"
            ))
    
    return text_blocks


def _extract_from_pdf_images(
    pdf_path: Path,
    page_numbers: List[int],
    language: str,
    engine: str,
    enable_vision: bool = False
) -> Tuple[List[TextBlock], str, Optional[str]]:
    """
    Convert PDF pages to images and extract text using OCR.
    
    Args:
        pdf_path: Path to PDF file
        page_numbers: List of page indices (0-based) to process
        language: Language code for OCR
        engine: OCR engine preference ("auto", "tesseract", "easyocr", "vision")
        enable_vision: Whether to enable Vision API as fallback
    
    Returns:
        Tuple of (list of TextBlock objects, engine name used, vision model name if used)
    """
    text_blocks: List[TextBlock] = []
    used_engine = "none"
    vision_model = None
    
    try:
        from pdf2image import convert_from_path
        logger.info("[OCR] PDF → image conversion")
    except ImportError:
        logger.debug("[OCR] pdf2image not available, cannot convert PDF pages to images")
        return [], "none", None
    
    # Detect if handwritten from filename (mirrors ENG-101-table-detector line 2121)
    filename_lower = pdf_path.name.lower()
    is_handwritten = any(keyword in filename_lower for keyword in ['handwritten', 'hand', 'scribble', 'notes'])
    
    try:
        # Convert PDF pages to images at ~300 DPI (mirrors ENG-101-table-detector line 1892)
        # This improves OCR quality for handwritten PDFs by ensuring sufficient resolution
        images = convert_from_path(str(pdf_path), first_page=min(page_numbers) + 1, last_page=max(page_numbers) + 1, dpi=300)
        
        for idx, page_num in enumerate(page_numbers):
            if idx >= len(images):
                continue
            
            img = images[idx]
            dimensions = img.size
            
            # Try OCR engines: For handwritten, EasyOCR FIRST (better for handwriting); otherwise Tesseract first
            # This mirrors ENG-101-table-detector behavior (line 1747-1768)
            page_blocks = []
            page_confidence = 0.0
            page_engine = "none"
            page_vision_model = None
            
            # For handwritten, try EasyOCR FIRST
            if is_handwritten and (engine == "auto" or engine == "easyocr"):
                try:
                    page_blocks, page_confidence = _extract_with_easyocr_image(img, language, dimensions)
                    if page_blocks:
                        page_engine = "easyocr"
                        logger.debug(f"[OCR] EasyOCR used for handwritten PDF page {page_num + 1} (tried first)")
                except Exception as e:
                    logger.debug(f"[OCR] EasyOCR failed for handwritten PDF page {page_num + 1}: {e}")
            
            # Try Tesseract (with PSM fallbacks for handwritten if EasyOCR didn't work)
            if not page_blocks and (engine == "auto" or engine == "tesseract"):
                try:
                    page_blocks, page_confidence = _extract_with_tesseract_image(img, language, dimensions, is_handwritten=is_handwritten)
                    if page_blocks:
                        page_engine = "tesseract"
                        logger.debug(f"[OCR] Tesseract used for PDF page {page_num + 1}")
                    else:
                        logger.debug(f"[OCR] Tesseract returned no blocks for PDF page {page_num + 1}")
                except Exception as e:
                    logger.debug(f"[OCR] Tesseract failed for page {page_num + 1}: {e}")
            
                # If handwritten and Tesseract failed or returned no blocks, try PSM fallbacks
                if is_handwritten and not page_blocks:
                    logger.debug(f"[OCR] Trying Tesseract PSM fallbacks for handwritten PDF page {page_num + 1}...")
                    for psm_mode in [11, 12, 13]:
                        try:
                            page_blocks, page_confidence = _extract_with_tesseract_image(
                                img, language, dimensions,
                                is_handwritten=is_handwritten,
                                psm_mode=psm_mode
                            )
                            if page_blocks:
                                page_engine = "tesseract"
                                logger.debug(f"[OCR] Tesseract PSM {psm_mode} succeeded for handwritten PDF page {page_num + 1}")
                                break
                        except Exception as e:
                            logger.debug(f"[OCR] Tesseract PSM {psm_mode} failed for PDF page {page_num + 1}: {e}")
            
            # If Tesseract didn't work and NOT handwritten, try EasyOCR
            if not page_blocks and not is_handwritten and (engine == "auto" or engine == "easyocr"):
                try:
                    page_blocks, page_confidence = _extract_with_easyocr_image(img, language, dimensions)
                    if page_blocks:
                        page_engine = "easyocr"
                        logger.debug(f"[OCR] EasyOCR used for PDF page {page_num + 1}")
                except Exception as e:
                    logger.debug(f"[OCR] EasyOCR failed for page {page_num + 1}: {e}")
            
            if not page_blocks and enable_vision and (engine == "auto" or engine == "vision"):
                try:
                    page_blocks, page_confidence, page_vision_model = _extract_with_vision_image(img, dimensions)
                    if page_blocks:
                        page_engine = "vision"
                        vision_model = page_vision_model
                        logger.debug(f"[OCR] Vision API used for PDF page {page_num + 1} (model={page_vision_model})")
                except Exception as e:
                    logger.debug(f"[OCR] Vision API failed for page {page_num + 1}: {e}")
            
            if not page_blocks:
                # Try fallback for this page
                logger.debug(f"[OCR] Fallback used for PDF page {page_num + 1}")
                page_engine = "fallback"
            
            # Track which engine was used (use first successful one)
            if page_blocks and used_engine == "none":
                used_engine = page_engine
            
            text_blocks.extend(page_blocks)
            
            if page_blocks:
                logger.debug(f"[OCR] Extracted {len(page_blocks)} text blocks from PDF page {page_num + 1}")
    
    except Exception as e:
        logger.warning(f"[OCR] Error converting PDF to images: {e}")
        return [], "none", None
    
    return text_blocks, used_engine, vision_model


def _extract_with_tesseract_image(
    img,
    language: str,
    dimensions: Optional[Tuple[int, int]],
    is_handwritten: bool = False,
    psm_mode: Optional[int] = None
) -> Tuple[List[TextBlock], float]:
    """
    Extract text from PIL Image using Tesseract OCR with preprocessing.
    
    Mirrors ENG-101-table-detector preprocessing behavior:
    - Handwritten: denoising + Otsu thresholding + light morphology
    - Printed: adaptive thresholding + dilation
    This improves OCR quality before extraction, addressing root cause of null fields.
    
    Args:
        img: PIL Image object
        language: Language code for OCR
        dimensions: Image dimensions (width, height)
        is_handwritten: Whether image contains handwritten text
        psm_mode: Tesseract PSM mode (if None, uses default; for handwritten, try 11, 12, 13)
    """
    import pytesseract
    from PIL import Image
    
    img_original = img.copy()
    img_processed = None
    
    # OCR PREPROCESSING (mirrors ENG-101-table-detector/src/extract_table_and_map_schema.py:1700-1741)
    # This improves OCR quality, which directly improves extraction accuracy for handwritten/PNG sources
    # If cv2/numpy not available, skip preprocessing and use original image
    try:
        import cv2
        import numpy as np
        
        # Convert to numpy array for preprocessing
        img_array = np.array(img_original)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        if is_handwritten:
            # Handwritten-specific preprocessing (ENG-101: _preprocess_for_handwritten, line 1445)
            # Denoise first (handwritten text benefits from noise reduction)
            img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
            
            # Use Otsu's threshold instead of adaptive for handwritten (often works better)
            _, img_thresh = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Light morphological operations to smooth without losing detail
            kernel = np.ones((2, 2), np.uint8)
            img_processed_array = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            img_processed = Image.fromarray(img_processed_array)
            logger.warning(f"[OCR] Applied handwritten preprocessing (denoise + Otsu + morphology)")
        else:
            # Standard preprocessing for printed text (ENG-101: line 1735-1741)
            img_thresh = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # Apply light dilation (kernel 1x2) to connect characters slightly
            kernel = np.ones((1, 2), np.uint8)
            img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)
            img_processed = Image.fromarray(img_dilated)
            logger.warning(f"[OCR] Applied printed text preprocessing (adaptive threshold + dilation)")
    except (ImportError, Exception) as e:
        # If preprocessing fails (cv2/numpy not available or processing error), use original image
        logger.warning(f"[OCR] Preprocessing skipped (cv2/numpy unavailable or error): {e}, using original image")
        img_processed = None
    
    # Use preprocessed image if available, otherwise original
    img = img_processed if img_processed is not None else img_original
    
    # Build Tesseract config string
    # Default PSM: 6 (uniform block of text) for printed, or use provided psm_mode
    if psm_mode is not None:
        tesseract_config = f"--oem 3 --psm {psm_mode}"
    else:
        tesseract_config = "--oem 3 --psm 6"  # Default for printed text
    
    # Get OCR data with bounding boxes
    try:
        ocr_data = pytesseract.image_to_data(img, lang=language, config=tesseract_config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        # If preprocessed image fails and we have original, try original (especially for handwritten)
        if img_processed is not None and is_handwritten:
            logger.debug(f"[OCR] Preprocessed image failed, trying original: {e}")
            ocr_data = pytesseract.image_to_data(img_original, lang=language, config=tesseract_config, output_type=pytesseract.Output.DICT)
        else:
            raise
    
    text_blocks = []
    current_line = 1
    last_y = None
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            conf = float(ocr_data['conf'][i]) / 100.0 if ocr_data['conf'][i] != -1 else 0.5
            x = float(ocr_data['left'][i])
            y = float(ocr_data['top'][i])
            w = float(ocr_data['width'][i])
            h = float(ocr_data['height'][i])
            
            # Determine line number
            line_num = None
            if last_y is None:
                line_num = current_line
                last_y = y
            elif abs(y - last_y) <= 10.0:
                line_num = current_line
            else:
                current_line += 1
                line_num = current_line
                last_y = y
            
            text_blocks.append(TextBlock(
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=conf,
                line_number=line_num,
                block_type="text"
            ))
    
    avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
    return text_blocks, avg_confidence


def _extract_with_easyocr_image(
    img,
    language: str,
    dimensions: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float]:
    """Extract text from PIL Image using EasyOCR."""
    import easyocr
    import numpy as np
    
    # Map language codes
    lang_map = {"eng": "en", "en": "en"}
    easyocr_lang = lang_map.get(language, "en")
    
    reader = easyocr.Reader([easyocr_lang])
    
    # Convert PIL Image to numpy array
    img_array = np.array(img)
    results = reader.readtext(img_array)
    
    text_blocks = []
    current_line = 1
    last_y = None
    
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Determine line number
        line_num = None
        if last_y is None:
            line_num = current_line
            last_y = y_min
        elif abs(y_min - last_y) <= 10.0:
            line_num = current_line
        else:
            current_line += 1
            line_num = current_line
            last_y = y_min
        
        text_blocks.append(TextBlock(
            text=text.strip(),
            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
            confidence=float(conf),
            line_number=line_num,
            block_type="text"
        ))
    
    avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
    return text_blocks, avg_confidence


def _extract_with_vision_image(
    img,
    dimensions: Optional[Tuple[int, int]]
) -> Tuple[List[TextBlock], float, Optional[str]]:
    """
    Extract text from PIL Image using OpenAI Vision API.
    
    Args:
        img: PIL Image object
        dimensions: Image dimensions (width, height)
    
    Returns:
        Tuple of (list of TextBlock objects, average confidence, vision model name)
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env file or environment variable.")
    
    try:
        from openai import OpenAI
        import base64
        import io
        import json
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Get image dimensions
        img_width, img_height = img.size if dimensions is None else dimensions
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Use IMAGE_MODEL from config/env, with fallback to gpt-4-turbo if configured model fails
        vision_model = IMAGE_MODEL
        fallback_model = "gpt-4-turbo"  # Generic fallback for vision-capable models
        logger.info(f"[OCR Vision] Using model: {vision_model} (fallback: {fallback_model})")
        
        try:
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract all text from this image with bounding boxes. 
Return a JSON array where each object has:
- "text": the extracted text string
- "bbox": [x_min, y_min, x_max, y_max] in pixels
- "confidence": confidence score 0.0-1.0

Image dimensions: {}x{} pixels.

Return ONLY valid JSON, no markdown formatting.""".format(img_width, img_height)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
        except Exception as e:
            # Fallback to gpt-4-turbo if configured model fails (e.g., model not available)
            logger.warning(f"[OCR Vision] Model {vision_model} failed: {e}, trying fallback {fallback_model}")
            if vision_model != fallback_model:
                vision_model = fallback_model
                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract all text from this image with bounding boxes. 
Return a JSON array where each object has:
- "text": the extracted text string
- "bbox": [x_min, y_min, x_max, y_max] in pixels
- "confidence": confidence score 0.0-1.0

Image dimensions: {}x{} pixels.

Return ONLY valid JSON, no markdown formatting.""".format(img_width, img_height)
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000
                )
            else:
                raise
        
        # Parse response
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Parse JSON
        try:
            blocks_data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                blocks_data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from Vision API response: {content[:200]}")
        
        # Convert to TextBlock objects
        text_blocks = []
        current_line = 1
        last_y = None
        
        for block_data in blocks_data:
            if not isinstance(block_data, dict):
                continue
            
            text = block_data.get("text", "").strip()
            if not text:
                continue
            
            bbox = block_data.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            conf = float(block_data.get("confidence", 0.9))
            
            # Determine line number
            line_num = None
            if last_y is None:
                line_num = current_line
                last_y = y_min
            elif abs(y_min - last_y) <= 10.0:
                line_num = current_line
            else:
                current_line += 1
                line_num = current_line
                last_y = y_min
            
            text_blocks.append(TextBlock(
                text=text,
                bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
                confidence=conf,
                line_number=line_num,
                block_type="text"
            ))
        
        avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
        logger.debug(f"[OCR] Vision API extracted {len(text_blocks)} text blocks")
        return text_blocks, avg_confidence, vision_model
    
    except ImportError:
        raise ValueError("openai package not installed. Install with: pip install openai")
    except Exception as e:
        logger.warning(f"[OCR] Vision API extraction failed: {e}")
        raise


def _extract_from_pdf_vision(
    pdf_path: Path,
    page_numbers: List[int],
    dimensions: Optional[Tuple[int, int]] = None
) -> Tuple[List[TextBlock], float, Optional[str]]:
    """
    Extract text from PDF pages using Vision API as fallback.
    
    Args:
        pdf_path: Path to PDF file
        page_numbers: List of page indices (0-based) to process
        dimensions: Optional image dimensions
    
    Returns:
        Tuple of (list of TextBlock objects, average confidence, vision model name)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ValueError("pdf2image not available, cannot convert PDF pages to images")
    
    text_blocks: List[TextBlock] = []
    vision_model = None
    
    try:
        # Convert PDF pages to images at ~300 DPI (mirrors ENG-101-table-detector line 1892)
        # This improves Vision API quality for handwritten PDFs by ensuring sufficient resolution
        images = convert_from_path(str(pdf_path), first_page=min(page_numbers) + 1, last_page=max(page_numbers) + 1, dpi=300)
        
        for idx, page_num in enumerate(page_numbers):
            if idx >= len(images):
                continue
            
            img = images[idx]
            page_blocks, page_conf, page_model = _extract_with_vision_image(img, img.size)
            
            if page_blocks:
                text_blocks.extend(page_blocks)
                if vision_model is None:
                    vision_model = page_model
                logger.debug(f"[OCR] Vision API extracted {len(page_blocks)} blocks from PDF page {page_num + 1}")
    
    except Exception as e:
        logger.warning(f"[OCR] Error in Vision API PDF extraction: {e}")
        return [], 0.0, None
    
    avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0.0
    return text_blocks, avg_confidence, vision_model


def _upscale_image_if_low_dpi(img: Image.Image, min_dpi: int = 200) -> Image.Image:
    """
    Upscale image if DPI is below threshold (mirrors ENG-101-table-detector DPI handling).
    
    For images with low DPI, upscaling improves OCR quality before extraction.
    This addresses root cause of null fields by ensuring sufficient resolution.
    
    Args:
        img: PIL Image to check/upscale
        min_dpi: Minimum DPI threshold (default: 200)
    
    Returns:
        Original or upscaled PIL Image
    """
    try:
        # Estimate DPI from image dimensions (rough heuristic)
        # For typical document sizes, assume 8.5x11 inches if dimensions suggest document
        width, height = img.size
        
        # Heuristic: if image is small (< 2000px width), likely low DPI
        # Upscale by factor to reach ~200 DPI equivalent
        if width < 2000:
            # Calculate upscale factor to reach ~200 DPI equivalent
            # Assuming original is ~72-100 DPI, upscale by ~2-3x
            scale_factor = max(2.0, 2000 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            logger.debug(f"[OCR] Upscaling image from {width}x{height} to {new_width}x{new_height} (estimated DPI improvement)")
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.debug(f"[OCR] Error upscaling image: {e}, using original")
    
    return img


def _get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions (width, height) in pixels.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (width, height) or None if unable to read
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except ImportError:
        logger.debug("PIL/Pillow not available for image dimension detection")
        return None
    except Exception as e:
        logger.warning(f"Failed to get image dimensions: {e}")
        return None

