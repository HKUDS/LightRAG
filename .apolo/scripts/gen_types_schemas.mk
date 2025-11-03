.PHONY: all clean test gen-types-schemas

all: gen-types-schemas

gen-types-schemas:
	@.apolo/scripts/gen_types_schemas.sh

clean:
	@rm -f .apolo/src/apolo_apps_lightrag/schemas/LightRAGAppInputs.json
	@rm -f .apolo/src/apolo_apps_lightrag/schemas/LightRAGAppOutputs.json
	@rm -f .apolo/src/apolo_apps_lightrag/types.py

test:
	@echo "No tests defined for schema generation."
