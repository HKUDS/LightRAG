# Prompts Changelog

All notable changes to the prompts system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2024-11-11

### 🎉 Initial Release - Prompts Externalization

Major refactoring to extract prompts from Python code to external Markdown files.

### Added

#### Files Structure
- Created `lightrag/prompts/` directory
- Added 10 main prompt files in Markdown format:
  - `entity_extraction_system_prompt.md`
  - `entity_extraction_user_prompt.md`
  - `entity_continue_extraction_user_prompt.md`
  - `summarize_entity_descriptions.md`
  - `fail_response.md`
  - `rag_response.md`
  - `naive_rag_response.md`
  - `kg_query_context.md`
  - `naive_query_context.md`
  - `keywords_extraction.md`

- Added 6 example files:
  - `entity_extraction_example_1.md` - Narrative text example
  - `entity_extraction_example_2.md` - Financial/market data example
  - `entity_extraction_example_3.md` - Sports event example
  - `keywords_extraction_example_1.md` - International trade example
  - `keywords_extraction_example_2.md` - Deforestation example
  - `keywords_extraction_example_3.md` - Education example

#### Documentation
- `lightrag/prompts/README.md` - Overview and usage guide
- `lightrag/prompts/DOCKER_USAGE.md` - Docker-specific usage guide
- `docs/PromptCustomization.md` - Complete customization guide

#### Docker Support
- Volume mapping in `docker-compose.yml` for prompts directory
- Development override config: `docker-compose.prompts-dev.yml`
- Updated `Dockerfile` with prompts directory support

#### Code Changes
- Added `_load_prompt_from_file()` function in `lightrag/prompt.py`
- Added `_load_examples_from_files()` function in `lightrag/prompt.py`
- Implemented dynamic prompt loading mechanism
- Reduced `prompt.py` from 422 lines to 88 lines (-79%)

### Changed

#### Format
- Changed prompt format from Python strings to Markdown (`.md`)
- UTF-8 encoding for all prompt files
- Consistent formatting across all prompts

#### Loading Mechanism
- Prompts now loaded dynamically from files at import time
- Examples loaded automatically with naming convention
- Error handling for missing prompt files

### Technical Details

#### Backward Compatibility
- ✅ PROMPTS dictionary structure unchanged
- ✅ All dictionary keys remain identical
- ✅ API unchanged - existing code works without modification
- ✅ Placeholder variables preserved

#### File Naming Convention
- Main prompts: `{name}.md`
- Examples: `{base_name}_{number}.md`
- All lowercase with underscores

#### Placeholders Preserved
All original placeholders maintained:
- `{entity_types}`, `{tuple_delimiter}`, `{completion_delimiter}`
- `{language}`, `{input_text}`, `{examples}`
- `{response_type}`, `{user_prompt}`, `{context_data}`
- `{description_type}`, `{description_name}`, `{description_list}`
- `{summary_length}`, `{query}`

### Performance

- No performance degradation
- File I/O only at module import time
- Caching via Python module system
- Minimal memory overhead

### Testing

- ✅ All 14 PROMPTS dictionary keys validated
- ✅ UTF-8 encoding verified
- ✅ Placeholder integrity confirmed
- ✅ Docker volume mounting tested
- ✅ No linter errors
- ✅ 100% backward compatibility verified

### Migration Guide

#### For End Users
No action required. Update will be transparent.

#### For Developers
```bash
git pull
# Prompts automatically load from new location
```

#### For Docker Users
```bash
git pull
docker-compose pull  # or rebuild
docker-compose up -d

# Optional: Enable live editing
# Add to docker-compose.yml:
# - ./lightrag/prompts:/app/lightrag/prompts
```

#### For Custom Deployments
Ensure `lightrag/prompts/` directory exists with all `.md` files.

### Benefits

#### Maintainability
- ✅ 79% reduction in `prompt.py` line count
- ✅ Clear separation of code and content
- ✅ Easier to review changes in git

#### User Experience
- ✅ Edit prompts without Python knowledge
- ✅ Use any text editor
- ✅ Markdown preview support
- ✅ Syntax highlighting

#### DevOps
- ✅ Docker volume mounting for live editing
- ✅ No image rebuild needed for prompt changes
- ✅ Different prompts per environment
- ✅ Easy A/B testing

### Known Issues

None at this time.

### Security

- No security implications
- Files read with UTF-8 encoding
- No user input in file paths
- Standard file I/O operations

### Dependencies

No new dependencies added.

### Breaking Changes

None. This release is 100% backward compatible.

---

## Future Versions

### Planned for [1.1.0]

#### Features Under Consideration
- [ ] Hot reload without application restart
- [ ] API endpoint to reload prompts
- [ ] File watcher for auto-reload
- [ ] Prompt validation tool
- [ ] Prompt versioning system

#### Enhancements
- [ ] Multi-language prompt variants
- [ ] Prompt A/B testing framework
- [ ] Performance metrics integration
- [ ] Quality monitoring dashboard

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-11-11 | Initial release - Prompts externalization |

---

## Contributing

When modifying prompts:

1. **Preserve placeholders** - Don't remove `{variable_name}`
2. **Test thoroughly** - Validate changes before committing
3. **Document changes** - Update this CHANGELOG
4. **Backup first** - Use git branches or backup files
5. **Version control** - Commit with clear messages

### Prompt Modification Checklist

- [ ] Placeholders intact
- [ ] UTF-8 encoding
- [ ] No syntax errors
- [ ] Tested with sample data
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Git committed

---

## Support

For questions or issues:

- Check [README.md](README.md) for basic usage
- Check [DOCKER_USAGE.md](DOCKER_USAGE.md) for Docker specifics
- Check [docs/PromptCustomization.md](../../docs/PromptCustomization.md) for advanced guide
- Open issue on GitHub
- Contact development team

---

**Maintained by:** LightRAG Team  
**Last Updated:** November 11, 2024

