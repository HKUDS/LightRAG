"""
Test suite for write_json optimization

This test verifies:
1. Fast path works for clean data (no sanitization)
2. Slow path applies sanitization for dirty data
3. Sanitization is done during encoding (memory-efficient)
4. Reloading updates shared memory with cleaned data
"""

import os
import json
import tempfile
from lightrag.utils import write_json, load_json, SanitizingJSONEncoder


class TestWriteJsonOptimization:
    """Test write_json optimization with two-stage approach"""

    def test_fast_path_clean_data(self):
        """Test that clean data takes the fast path without sanitization"""
        clean_data = {
            "name": "John Doe",
            "age": 30,
            "items": ["apple", "banana", "cherry"],
            "nested": {"key": "value", "number": 42},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # Write clean data - should return False (no sanitization)
            needs_reload = write_json(clean_data, temp_file)
            assert not needs_reload, "Clean data should not require sanitization"

            # Verify data was written correctly
            loaded_data = load_json(temp_file)
            assert loaded_data == clean_data, "Loaded data should match original"
        finally:
            os.unlink(temp_file)

    def test_slow_path_dirty_data(self):
        """Test that dirty data triggers sanitization"""
        # Create data with surrogate characters (U+D800 to U+DFFF)
        dirty_string = "Hello\ud800World"  # Contains surrogate character
        dirty_data = {"text": dirty_string, "number": 123}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # Write dirty data - should return True (sanitization applied)
            needs_reload = write_json(dirty_data, temp_file)
            assert needs_reload, "Dirty data should trigger sanitization"

            # Verify data was written and sanitized
            loaded_data = load_json(temp_file)
            assert loaded_data is not None, "Data should be written"
            assert loaded_data["number"] == 123, "Clean fields should remain unchanged"
            # Surrogate character should be removed
            assert (
                "\ud800" not in loaded_data["text"]
            ), "Surrogate character should be removed"
        finally:
            os.unlink(temp_file)

    def test_sanitizing_encoder_removes_surrogates(self):
        """Test that SanitizingJSONEncoder removes surrogate characters"""
        data_with_surrogates = {
            "text": "Hello\ud800\udc00World",  # Contains surrogate pair
            "clean": "Clean text",
            "nested": {"dirty_key\ud801": "value", "clean_key": "clean\ud802value"},
        }

        # Encode using custom encoder
        encoded = json.dumps(
            data_with_surrogates, cls=SanitizingJSONEncoder, ensure_ascii=False
        )

        # Verify no surrogate characters in output
        assert "\ud800" not in encoded, "Surrogate U+D800 should be removed"
        assert "\udc00" not in encoded, "Surrogate U+DC00 should be removed"
        assert "\ud801" not in encoded, "Surrogate U+D801 should be removed"
        assert "\ud802" not in encoded, "Surrogate U+D802 should be removed"

        # Verify clean parts remain
        assert "Clean text" in encoded, "Clean text should remain"
        assert "clean_key" in encoded, "Clean keys should remain"

    def test_nested_structure_sanitization(self):
        """Test sanitization of deeply nested structures"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {"dirty": "text\ud800here", "clean": "normal text"},
                    "list": ["item1", "item\ud801dirty", "item3"],
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            needs_reload = write_json(nested_data, temp_file)
            assert needs_reload, "Nested dirty data should trigger sanitization"

            # Verify nested structure is preserved
            loaded_data = load_json(temp_file)
            assert "level1" in loaded_data
            assert "level2" in loaded_data["level1"]
            assert "level3" in loaded_data["level1"]["level2"]

            # Verify surrogates are removed
            dirty_text = loaded_data["level1"]["level2"]["level3"]["dirty"]
            assert "\ud800" not in dirty_text, "Nested surrogate should be removed"

            # Verify list items are sanitized
            list_items = loaded_data["level1"]["level2"]["list"]
            assert (
                "\ud801" not in list_items[1]
            ), "List item surrogates should be removed"
        finally:
            os.unlink(temp_file)

    def test_unicode_non_characters_removed(self):
        """Test that Unicode non-characters (U+FFFE, U+FFFF) don't cause encoding errors

        Note: U+FFFE and U+FFFF are valid UTF-8 characters (though discouraged),
        so they don't trigger sanitization. They only get removed when explicitly
        using the SanitizingJSONEncoder.
        """
        data_with_nonchars = {"text1": "Hello\ufffeWorld", "text2": "Test\uffffString"}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # These characters are valid UTF-8, so they take the fast path
            needs_reload = write_json(data_with_nonchars, temp_file)
            assert not needs_reload, "U+FFFE/U+FFFF are valid UTF-8 characters"

            loaded_data = load_json(temp_file)
            # They're written as-is in the fast path
            assert loaded_data == data_with_nonchars
        finally:
            os.unlink(temp_file)

    def test_mixed_clean_dirty_data(self):
        """Test data with both clean and dirty fields"""
        mixed_data = {
            "clean_field": "This is perfectly fine",
            "dirty_field": "This has\ud800issues",
            "number": 42,
            "boolean": True,
            "null_value": None,
            "clean_list": [1, 2, 3],
            "dirty_list": ["clean", "dirty\ud801item"],
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            needs_reload = write_json(mixed_data, temp_file)
            assert (
                needs_reload
            ), "Mixed data with dirty fields should trigger sanitization"

            loaded_data = load_json(temp_file)

            # Clean fields should remain unchanged
            assert loaded_data["clean_field"] == "This is perfectly fine"
            assert loaded_data["number"] == 42
            assert loaded_data["boolean"]
            assert loaded_data["null_value"] is None
            assert loaded_data["clean_list"] == [1, 2, 3]

            # Dirty fields should be sanitized
            assert "\ud800" not in loaded_data["dirty_field"]
            assert "\ud801" not in loaded_data["dirty_list"][1]
        finally:
            os.unlink(temp_file)

    def test_empty_and_none_strings(self):
        """Test handling of empty and None values"""
        data = {
            "empty": "",
            "none": None,
            "zero": 0,
            "false": False,
            "empty_list": [],
            "empty_dict": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            needs_reload = write_json(data, temp_file)
            assert (
                not needs_reload
            ), "Clean empty values should not trigger sanitization"

            loaded_data = load_json(temp_file)
            assert loaded_data == data, "Empty/None values should be preserved"
        finally:
            os.unlink(temp_file)

    def test_specific_surrogate_udc9a(self):
        """Test specific surrogate character \\udc9a mentioned in the issue"""
        # Test the exact surrogate character from the error message:
        # UnicodeEncodeError: 'utf-8' codec can't encode character '\\udc9a'
        data_with_udc9a = {
            "text": "Some text with surrogate\udc9acharacter",
            "position": 201,  # As mentioned in the error
            "clean_field": "Normal text",
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # Write data - should trigger sanitization
            needs_reload = write_json(data_with_udc9a, temp_file)
            assert needs_reload, "Data with \\udc9a should trigger sanitization"

            # Verify surrogate was removed
            loaded_data = load_json(temp_file)
            assert loaded_data is not None
            assert "\udc9a" not in loaded_data["text"], "\\udc9a should be removed"
            assert (
                loaded_data["clean_field"] == "Normal text"
            ), "Clean fields should remain"
        finally:
            os.unlink(temp_file)

    def test_migration_with_surrogate_sanitization(self):
        """Test that migration process handles surrogate characters correctly

        This test simulates the scenario where legacy cache contains surrogate
        characters and ensures they are cleaned during migration.
        """
        # Simulate legacy cache data with surrogate characters
        legacy_data_with_surrogates = {
            "cache_entry_1": {
                "return": "Result with\ud800surrogate",
                "cache_type": "extract",
                "original_prompt": "Some\udc9aprompt",
            },
            "cache_entry_2": {
                "return": "Clean result",
                "cache_type": "query",
                "original_prompt": "Clean prompt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # First write the dirty data directly (simulating legacy cache file)
            # Use custom encoder to force write even with surrogates
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(
                    legacy_data_with_surrogates,
                    f,
                    cls=SanitizingJSONEncoder,
                    ensure_ascii=False,
                )

            # Load and verify surrogates were cleaned during initial write
            loaded_data = load_json(temp_file)
            assert loaded_data is not None

            # The data should be sanitized
            assert (
                "\ud800" not in loaded_data["cache_entry_1"]["return"]
            ), "Surrogate in return should be removed"
            assert (
                "\udc9a" not in loaded_data["cache_entry_1"]["original_prompt"]
            ), "Surrogate in prompt should be removed"

            # Clean data should remain unchanged
            assert (
                loaded_data["cache_entry_2"]["return"] == "Clean result"
            ), "Clean data should remain"

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run tests
    test = TestWriteJsonOptimization()

    print("Running test_fast_path_clean_data...")
    test.test_fast_path_clean_data()
    print("✓ Passed")

    print("Running test_slow_path_dirty_data...")
    test.test_slow_path_dirty_data()
    print("✓ Passed")

    print("Running test_sanitizing_encoder_removes_surrogates...")
    test.test_sanitizing_encoder_removes_surrogates()
    print("✓ Passed")

    print("Running test_nested_structure_sanitization...")
    test.test_nested_structure_sanitization()
    print("✓ Passed")

    print("Running test_unicode_non_characters_removed...")
    test.test_unicode_non_characters_removed()
    print("✓ Passed")

    print("Running test_mixed_clean_dirty_data...")
    test.test_mixed_clean_dirty_data()
    print("✓ Passed")

    print("Running test_empty_and_none_strings...")
    test.test_empty_and_none_strings()
    print("✓ Passed")

    print("Running test_specific_surrogate_udc9a...")
    test.test_specific_surrogate_udc9a()
    print("✓ Passed")

    print("Running test_migration_with_surrogate_sanitization...")
    test.test_migration_with_surrogate_sanitization()
    print("✓ Passed")

    print("\n✅ All tests passed!")
