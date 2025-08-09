# 🐛 Issue Report for Docling GitHub Repository

## Bug Report Template

**Repository:** https://github.com/docling-project/docling/issues

### Title
```
Pydantic ValidationError when DOCLING_DEBUG environment variable is set to boolean false
```

### Description

**Summary:**
When the environment variable `DOCLING_DEBUG=false` is set, Docling fails to initialize with a Pydantic validation error during the import of `DocumentConverter`. This prevents the library from functioning entirely.

**Expected Behavior:**
The library should either:
1. Accept boolean values for the `debug` configuration
2. Ignore unknown environment variables that don't match its configuration schema
3. Provide clear documentation about expected environment variable formats

**Actual Behavior:**
The library crashes with a Pydantic validation error during initialization.

### Error Details

**Error Message:**
```
ValidationError: 1 validation error for AppSettings
debug
  Input should be a valid dictionary or instance of DebugSettings [type=model_type, input_value=False, input_type=bool]
    For further information visit https://errors.pydantic.dev/2.11/v/model_type
```

**Full Stack Trace:**
```python
Traceback (most recent call last):
  File "<string>", line 3, in <module>
  File "/home/docling/.local/lib/python3.11/site-packages/docling/document_converter.py", line 15, in <module>
    from docling.backend.asciidoc_backend import AsciiDocBackend
  File "/home/docling/.local/lib/python3.11/site-packages/docling/backend/asciidocbackend.py", line 21, in <module>
    from docling.datamodel.document import InputDocument
  File "/home/docling/.local/lib/python3.11/site-packages/docling/datamodel/document.py", line 69, in <module>
    from docling.datamodel.settings import DocumentLimits
  File "/home/docling/.local/lib/python3.11/site-packages/docling/datamodel/settings.py", line 65, in <module>
    settings = AppSettings()
    ^^^^^^^^^^^^^
  File "/home/docling/.local/lib/python3.11/site-packages/pydantic_settings/main.py", line 188, in __init__
    super().__init__(
  File "/home/docling/.local/lib/python3.11/site-packages/pydantic/main.py", line 253, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for AppSettings
debug
  Input should be a valid dictionary or instance of DebugSettings [type=model_type, input_value=False, input_type=bool]
    For further information visit https://errors.pydantic.dev/2.11/v/model_type
```

### Environment

**System Information:**
- **OS:** Linux (Docker container)
- **Python Version:** 3.11
- **Docling Version:** [Check your installed version]
- **Pydantic Version:** 2.11+ (based on error message)
- **Installation Method:** pip install docling

**Environment Variables:**
```bash
DOCLING_DEBUG=false  # This causes the error
```

### Steps to Reproduce

1. Set environment variable: `export DOCLING_DEBUG=false`
2. Try to import DocumentConverter:
   ```python
   from docling.document_converter import DocumentConverter
   converter = DocumentConverter()
   ```
3. Observe the Pydantic validation error

### Minimal Reproduction Case

```python
import os
os.environ['DOCLING_DEBUG'] = 'false'  # Set as string 'false'

# This will fail with Pydantic validation error
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
```

### Workaround

**Current Solution:**
Remove the `DOCLING_DEBUG` environment variable entirely:
```bash
unset DOCLING_DEBUG
```

### Root Cause Analysis

The issue appears to be in `docling/datamodel/settings.py` where `AppSettings()` is initialized. The Pydantic model expects the `debug` field to be either:
- A dictionary
- An instance of `DebugSettings` class

But when `DOCLING_DEBUG=false` is set as an environment variable, it's being interpreted as a boolean `False`, which doesn't match the expected type.

### Suggested Fixes

**Option 1: Support Boolean Values**
Modify the Pydantic model to accept boolean values for debug configuration:
```python
debug: Union[bool, Dict, DebugSettings] = False
```

**Option 2: Environment Variable Validation**
Add validation to handle string representations of boolean values:
```python
@field_validator('debug', mode='before')
@classmethod
def validate_debug(cls, v):
    if isinstance(v, str):
        if v.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif v.lower() in ('false', '0', 'no', 'off'):
            return False
    return v
```

**Option 3: Documentation**
Clearly document the expected format for environment variables in the configuration documentation.

### Impact

**Severity:** High
- **User Impact:** Complete failure to initialize Docling when certain environment variables are set
- **Frequency:** Likely affects users integrating Docling into containerized environments where environment variables are commonly used for configuration

### Additional Context

This issue was discovered while integrating Docling into a microservice architecture where environment variables are commonly used for configuration management. The boolean value `false` is a natural choice for disabling debug mode, but causes complete library failure.

### Related Files

Based on the stack trace, the issue is likely in:
- `docling/datamodel/settings.py` (line 65)
- The `AppSettings` class definition
- Potentially the `DebugSettings` class definition

### Environment Variable Documentation

It would be helpful to have documentation clarifying:
1. Which environment variables are supported by Docling
2. Expected formats for each environment variable
3. Default values when environment variables are not set

### Testing Suggestion

Consider adding test cases for:
1. Various environment variable configurations
2. Edge cases like boolean string values
3. Invalid environment variable formats

---

**Labels to add:** `bug`, `pydantic`, `configuration`, `environment-variables`
**Priority:** High (blocks library initialization)
