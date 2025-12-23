# Indian License Plate Normalization Utility 🚗

A comprehensive Python utility for normalizing, cleaning, and validating Indian license plate text extracted from OCR systems.

## Features ✨

- 🧹 **Text Normalization**: Remove noise, fix spacing, convert to proper case
- 🔍 **Format Detection**: Automatically detect different Indian plate formats
- ✅ **Validation**: Validate against Indian license plate patterns and rules
- 🏛️ **State Recognition**: Identify and validate Indian state codes
- 🔧 **OCR Error Correction**: Handle common OCR misreadings
- 📊 **Confidence Scoring**: Reliability metrics for validation results
- 🚀 **Easy Integration**: Simple API for ANPR systems

## Supported Formats 🏷️

| Format Type | Example | Description |
|-------------|---------|-------------|
| **Standard New** | `DL01AB1234` | Current format without spaces |
| **Standard Old** | `DL 01 AB 1234` | Traditional format with spaces |
| **Bharat Series** | `22 BH 1234 AB` | New unified numbering system |
| **Temporary** | `T1234` | Temporary registration plates |

## Quick Start 🚀

### Basic Usage

```python
from vision.utils.plate_normalizer import normalize_license_plate

# Normalize noisy OCR output
result = normalize_license_plate("dl@01#ab$1234")

print(f"Valid: {result.is_valid}")           # True
print(f"Normalized: {result.normalized_text}") # "DL 01 AB 1234"
print(f"State: {result.state_code}")          # "DL"
print(f"Confidence: {result.confidence_score}") # 1.000
```

### Utility Functions

```python
from vision.utils.plate_normalizer import (
    is_valid_indian_plate,
    get_state_name,
    format_plate_display
)

# Quick validation
is_valid_indian_plate("DL01AB1234")  # True

# Get state name
get_state_name("MH")  # "Maharashtra"

# Format for display
format_plate_display("dl01ab1234")  # "DL 01 AB 1234"
```

## API Reference 📚

### Main Function

#### `normalize_license_plate(text: str) -> PlateValidationResult`

Normalize and validate license plate text.

**Parameters:**
- `text` (str): Raw license plate text from OCR

**Returns:**
- `PlateValidationResult`: Comprehensive validation result

**Example:**
```python
result = normalize_license_plate("MH12DE3456")
# result.is_valid = True
# result.normalized_text = "MH 12 DE 3456"  
# result.state_code = "MH"
# result.district_code = "12"
# result.series = "DE"
# result.number = "3456"
```

### PlateValidationResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_valid` | bool | Whether the plate is valid |
| `normalized_text` | str | Cleaned and formatted text |
| `original_text` | str | Original input text |
| `format_type` | PlateFormat | Detected format type |
| `state_code` | str | Two-letter state code |
| `district_code` | str | District registration code |
| `series` | str | Letter series |
| `number` | str | Number portion |
| `confidence_score` | float | Validation confidence (0.0-1.0) |
| `errors` | List[str] | Validation error messages |

### Utility Functions

#### `is_valid_indian_plate(text: str) -> bool`
Quick validation check.

#### `get_state_name(state_code: str) -> Optional[str]`
Get full state name from code.

#### `format_plate_display(text: str) -> str`
Format plate for display with proper spacing.

## OCR Integration 🔗

Perfect for enhancing ANPR systems:

```python
def process_ocr_result(raw_ocr_text, ocr_confidence):
    """Process OCR output in ANPR system."""
    result = normalize_license_plate(raw_ocr_text)
    
    if result.is_valid and result.confidence_score > 0.7:
        # High confidence - accept result
        return {
            'status': 'ACCEPT',
            'plate': result.normalized_text,
            'state': get_state_name(result.state_code),
            'confidence': result.confidence_score
        }
    elif result.is_valid and result.confidence_score > 0.4:
        # Medium confidence - needs review
        return {'status': 'REVIEW', 'plate': result.normalized_text}
    else:
        # Low confidence - reject
        return {'status': 'REJECT', 'errors': result.errors}
```

## Supported States 🏛️

All Indian states and union territories:

| Code | State/UT | Code | State/UT |
|------|----------|------|----------|
| AN | Andaman and Nicobar | MH | Maharashtra |
| AP | Andhra Pradesh | MN | Manipur |
| AR | Arunachal Pradesh | ML | Meghalaya |
| AS | Assam | MZ | Mizoram |
| BR | Bihar | NL | Nagaland |
| CH | Chandigarh | OR | Odisha |
| CG | Chhattisgarh | PY | Puducherry |
| DL | Delhi | PB | Punjab |
| GA | Goa | RJ | Rajasthan |
| GJ | Gujarat | SK | Sikkim |
| HR | Haryana | TN | Tamil Nadu |
| HP | Himachal Pradesh | TG | Telangana |
| JK | Jammu and Kashmir | TR | Tripura |
| JH | Jharkhand | UP | Uttar Pradesh |
| KA | Karnataka | UK | Uttarakhand |
| KL | Kerala | WB | West Bengal |
| MP | Madhya Pradesh | BH | Bharat Series |

## Error Handling 🛠️

The utility handles common OCR errors:

```python
# OCR misreadings
normalize_license_plate("DL O1 AB 1234")  # O instead of 0
normalize_license_plate("MHI2DE3456")     # I instead of 1
normalize_license_plate("DL@01#AB$1234")  # Special characters

# All return properly normalized results or detailed error messages
```

## Testing 🧪

Run the comprehensive demo:

```bash
python demo_plate_normalizer.py
```

This demonstrates:
- Basic normalization
- OCR error correction
- Format detection
- Utility functions
- Integration examples

## Performance 📊

- ⚡ **Fast**: Processes plates in milliseconds
- 🎯 **Accurate**: High validation accuracy for Indian formats
- 🛡️ **Robust**: Handles noisy OCR input gracefully
- 📈 **Scalable**: Suitable for high-volume processing

## Integration Examples 🔌

### With Tesseract OCR

```python
import pytesseract
from vision.utils.plate_normalizer import normalize_license_plate

# OCR + Normalization pipeline
raw_text = pytesseract.image_to_string(plate_image)
result = normalize_license_plate(raw_text)

if result.is_valid:
    print(f"Detected plate: {result.normalized_text}")
    print(f"Vehicle from: {get_state_name(result.state_code)}")
```

### With YOLO + OCR

```python
def process_detected_plate(plate_crop):
    """Process a detected license plate crop."""
    # Step 1: OCR
    raw_text = ocr_engine.extract_text(plate_crop)
    
    # Step 2: Normalize and validate
    result = normalize_license_plate(raw_text)
    
    # Step 3: Return structured data
    return {
        'raw_ocr': raw_text,
        'normalized': result.normalized_text,
        'valid': result.is_valid,
        'state': result.state_code,
        'confidence': result.confidence_score,
        'components': {
            'state': result.state_code,
            'district': result.district_code,
            'series': result.series,
            'number': result.number
        }
    }
```

## Advanced Features 🔬

### Custom Normalization

```python
from vision.utils.plate_normalizer import IndianPlateNormalizer

# Create custom normalizer
normalizer = IndianPlateNormalizer()

# Step-by-step processing
cleaned = normalizer.remove_noise("DL@01#AB$1234")
spaced = normalizer.normalize_spacing(cleaned)
format_type = normalizer.detect_format(spaced)
components = normalizer.parse_plate_components(spaced, format_type)
```

### Confidence Thresholds

```python
result = normalize_license_plate(ocr_text)

if result.confidence_score > 0.9:
    status = "HIGH_CONFIDENCE"
elif result.confidence_score > 0.6:
    status = "MEDIUM_CONFIDENCE"
else:
    status = "LOW_CONFIDENCE"
```

## Contributing 🤝

This utility is part of the ANPR system. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License 📄

Part of the ANPR System project.

---

**💡 Pro Tip**: Use this utility in your ANPR pipeline to significantly improve license plate recognition accuracy and handle real-world OCR noise effectively!

**🚀 Ready to integrate?** Check out `demo_plate_normalizer.py` for comprehensive examples and best practices.
